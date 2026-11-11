"""TREE speculative decoding engine (research): Generator subclass running best-first draft
trees over the DSpine lattice. Single-job, greedy; chain path untouched (super used whenever
tree mode is off). Activated by Generator(..., tree_cap = N) with N >= 4."""
from __future__ import annotations
import os, time
import torch
from .generator import Generator
from ..constants import PAGE_SIZE
from ..ext import exllamav3_ext as ext


class TreeGenerator(Generator):

    def __init__(self, *args, tree_cap: int = 31, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree_cap = tree_cap
        self._tree = None            # (tokens [1, W+1] cpu, parents [W+1], depths [W+1], anc [W+1, W+1] int8 dev)

    # ---------------- draft producer ----------------

    def iterate_draftmodel_dflash_gen(self, results: list, window: int):
        if os.environ.get("DSPINE_TREE_DEBUG"):
            print(f"[tree-prod] called tree_cap={self.tree_cap} prefill={sum(1 for j in self.active_jobs if j.is_prefill_done())}", flush = True)
        if self.tree_cap <= 0 or not hasattr(self.draft_model, "sample_tree_from_state"):
            self._tree = None
            return super().iterate_draftmodel_dflash_gen(results, window)

        if os.environ.get("DSPINE_TREE_LINEAR"):
            # PROBE: linear-chain tree (parents = chain, mask = causal) must reproduce the
            # chain engine's output byte-exactly if the tree math is correct. Uses the chain
            # producer's own draft (single drafter forward, state stays aligned).
            chain = super().iterate_draftmodel_dflash_gen(results, window)
            if os.environ.get("DSPINE_TREE_DEBUG"):
                print(f"[tree-linear] super returned {None if chain is None else chain.shape}", flush = True)
            if chain is None:
                self._tree = None
                return None
            job = next((j for j in self.active_jobs if j.is_prefill_done()), None)
            anchor = int(job.sequences[0].sequence_ids.torch_slice(job.sequences[0].kv_position, None)[0, -1].item()) if job else int(chain[0, 0].item())
            ct = chain[0].tolist()
            W = len(ct)
            toks = torch.tensor([[anchor] + ct], dtype = torch.long)
            par = torch.tensor([-1] + list(range(W)), dtype = torch.int32, device = "cuda:0")
            dep = torch.tensor(list(range(W + 1)), dtype = torch.int32, device = "cuda:0")
            anc = torch.tril(torch.ones((W + 1, W + 1), dtype = torch.int8, device = par.device))
            self._tree = (toks, par, dep, anc.contiguous())
            par_h = [-1] + list(range(W))
            dep_h = list(range(W + 1))
            self._prep_tables(par_h, dep_h, par.device)
            if self.draft_ids_pinned.shape[-1] < W:
                self.draft_ids_pinned = torch.empty((max(32, self.draft_ids_pinned.shape[0]), W), dtype = torch.long, pin_memory = True)
            self.draft_ids_pinned[0, :W].copy_(toks[0, 1:W + 1].flatten())
            return self.draft_ids_pinned[:, :W]

        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(
                max_seq_len,
                job.get_max_seq_len() + self.tree_cap + 2,
                job.get_max_seq_len() + self.draft_model.config.block_size + 1,
            )
            batch_size += 1
        if batch_size == 0:
            return None
        max_pages_batch = ((max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE + 15) // 16 * 16
        block_index = self._staging("draft_block_index", batch_size, max_pages_batch)
        block_index.zero_()
        cache_seqlens = self._staging("draft_cache_seqlens", batch_size)
        batch = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            for seq in job.sequences:
                seq_bi = seq.block_index_tensor[:, :max_pages_batch]
                block_index[batch:batch + 1, :seq_bi.shape[-1]].copy_(seq_bi)
                cache_seqlens[batch] = seq.kv_position
                batch += 1
        input_ids_list = []
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            if job.time_first_token is None:
                from ..util import cuda_sync_active
                cuda_sync_active()
                job.time_first_token = time.time()
            input_ids_list += job.get_input_ids_list()
        batch_ids = self.draft_input_ids_pinned[:batch_size, :]
        batch_ids.copy_(torch.cat(input_ids_list, dim = 0))

        params = {
            "attn_mode": "flash_attn",
            "block_table": block_index,
            "cache": self.draft_cache,
            "cache_seqlens": cache_seqlens,
        }
        if self.draft_recurrent:
            params["recurrent_states"] = [job.recurrent_state.draft for job in self.active_jobs if job.is_prefill_done()]
        out_state = self.draft_model.forward(input_ids = batch_ids, params = params)

        toks, parents, depths = self.draft_model.sample_tree_from_state(out_state, params, cap = self.tree_cap)
        W = int((toks[0, 1:] >= 0).sum())          # drafted nodes
        toks = toks[:, :W + 1]                      # [1, W+1] cpu long
        par_h = parents[0, :W + 1].tolist()          # host copy: per-round topology tables
        dep_h = depths[:W + 1].tolist()
        par = parents[0, :W + 1].to("cuda:0", torch.int32)
        dep = depths[:W + 1].to("cuda:0", torch.int32)
        anc = torch.eye(W + 1, dtype = torch.int8, device = par.device)
        pl = par.tolist()
        for r in range(1, W + 1):
            p = pl[r]
            if p >= 0:
                anc[r] |= anc[p]
        self._tree = (toks, par, dep, anc.contiguous())
        self._prep_tables(par_h, dep_h, par.device)

        if self.draft_ids_pinned.shape[-1] < W:
            self.draft_ids_pinned = torch.empty((max(batch_size, 32), W), dtype = torch.long, pin_memory = True)
        self.draft_ids_pinned[:batch_size, :W].copy_(toks[:, 1:])
        return self.draft_ids_pinned[:, :W]

    def _prep_tables(self, par_h: list, dep_h: list, device):
        """Per-round topology tables built ONCE on host + one upload, consumed by every layer."""
        W1 = len(par_h)
        # levels: (idx_t, par_t) per depth level for the batched delta rule
        _lv: list[list[int]] = [[] for _ in range(W1 + 1)]
        for r in range(W1):
            _lv[dep_h[r]].append(r)
        self._tree_levels = [
            (torch.tensor(ix, dtype = torch.long, device = device),
             torch.tensor([par_h[r] for r in ix], dtype = torch.long, device = device))
            for ix in _lv if ix
        ]
        # conv index: [W1, K] row index per (row, col); negative = -1-ringcol for ring cols
        K_c = 4
        A = torch.full((W1, K_c), -1, dtype = torch.long)
        for r in range(W1):
            d = dep_h[r]
            chain = []
            p = par_h[r]
            while p >= 0:
                chain.append(p)
                p = par_h[p]
            for j in range(K_c):
                eff = d - (K_c - 1 - j)
                if eff == d: A[r, j] = r
                elif eff >= 0: A[r, j] = chain[d - 1 - eff]
                else: A[r, j] = -1 - (K_c + eff)
        self._tree_conv_index = A.to(device)
        self._tree_host = (par_h, dep_h)

    # ---------------- tree verify + commit ----------------

    def iterate_gen(self, results: list, draft_tokens: torch.Tensor | None = None):
        if draft_tokens is None or self._tree is None:
            self._tree = None
            return super().iterate_gen(results, draft_tokens)

        toks, par, dep, anc = self._tree
        self._tree = None
        _dbg = bool(os.environ.get("DSPINE_TREE_DEBUG"))
        _t0 = time.time()
        W = toks.shape[-1] - 1                     # drafted nodes; window rows = W + 1
        assert len(self.active_jobs) == 1, "tree mode: single job"
        job = next(j for j in self.active_jobs if j.is_prefill_done())
        assert len(job.sequences) == 1 and not any(f.is_active for f in job.filters), "tree mode: plain jobs"
        seq = job.sequences[0]
        kvp = seq.kv_position

        # Stage ids + ALLOCATE cache pages for anchor + tree rows (fills page.sequence,
        # advances per-page kv_position; seq.kv_position itself stays at kvp until commit)
        input_ids_list = job.get_input_ids_list(toks[:, 1:], 0, add_to_cache = True)
        assert input_ids_list[0].shape[-1] == W + 1, \
            f"tree window must be anchor+{W}, got {input_ids_list[0].shape[-1]}"

        max_seq_len = job.get_max_seq_len() + W + 16
        max_pages_batch = ((max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE + 15) // 16 * 16
        use_offsets = "mrope" in self.model.caps
        arena = self._stage_arena(1, max_pages_batch, use_offsets)
        block_index, cache_seqlens, positions = arena[1]
        seq_rope = [job.alt_rope_offset] if use_offsets else []
        batch_ids = torch.cat(input_ids_list, dim = 0)
        # Chain reference (generator.py iterate_gen): the general path passes batch_ids = None
        # to generator_stage_decode (tables only); ids reach the cache via get_input_ids_list
        ext.generator_stage_decode(arena[0], max_pages_batch, [seq.block_index_tensor], [kvp], seq_rope,
                                   None, [], [], [])

        batch_states = [job.recurrent_state.target] if (self.target_recurrent and job.recurrent_state) else None
        params = {
            "attn_mode": "flash_attn",
            "block_table": block_index,
            "cache": self.cache,
            "cache_seqlens": cache_seqlens,
            "recurrent_states": batch_states,
            "indexed_embeddings": job.embeddings,
            "positions": positions,
            "recurrent_history": True,           # chain reference: draft windows stage scan history
            "pinned_staging": True,
        }
        if not os.environ.get("DSPINE_TREE_NOATTN"):
            params["tree_parents"] = par
            params["tree_depths"] = dep
            params["tree_mask"] = anc
            params["tree_levels"] = getattr(self, "_tree_levels", None)      # precomputed (idx, par) per level
            params["tree_conv_index"] = getattr(self, "_tree_conv_index", None)
        self._upload_arena(arena, 1, max_pages_batch, use_offsets, params)
        if not os.environ.get("DSPINE_TREE_NOATTN"):
            pos_ids = (kvp + dep).view(1, W + 1)
            params["positions"] = None                      # tree rows are non-contiguous: ids only
            params["position_ids"] = pos_ids.to(torch.int32)
        if self.draft_model:
            params.update(self.draft_model.draft_verifier_params)
        batch_logits = self.model.forward(input_ids = batch_ids.to(anc.device), params = params)
        p_export_states = params.get("export_states")
        p_cache_seqlens = params.get("cache_seqlens")
        tree_out = params.get("tree_out") or {}
        params = None
        if os.environ.get("DSPINE_TREE_DEBUG"):
            from ..util import cuda_sync_active
            cuda_sync_active(); print(f"[tree] forward done {time.time()-_t0:.1f}s", flush = True)
        arg = batch_logits[0].argmax(dim = -1).tolist()
        if _dbg:
            print(f"[tree] R window={toks[0, :6].tolist()} anchor_arg={arg[0]} next_args={arg[1:4]}", flush = True)
        pl = par.tolist()
        dl = dep.tolist()
        accepted = [False] * (W + 1)
        accepted[0] = True
        best_row, best_depth = 0, 0
        for r in range(1, W + 1):
            p = pl[r]
            if p >= 0 and accepted[p] and int(toks[0, r]) == arg[p]:
                accepted[r] = True
                if dl[r] > best_depth:
                    best_row, best_depth = r, dl[r]
        path = [best_row]
        while pl[path[-1]] >= 0:                    # >= 0: include row 0 (its parent is -1)
            path.append(pl[path[-1]])
        path.reverse()                              # root .. best (row 0 first)
        acc_nodes = len(path) - 1
        n_keep = 1 + acc_nodes
        keep = torch.tensor(path, dtype = torch.long, device = batch_logits.device)

        # ---- KV compaction (attn layers) ----
        if _dbg:
            from ..util import cuda_sync_active; cuda_sync_active()
            print(f"[tree] walk done {time.time()-_t0:.2f}s acc={acc_nodes}", flush = True)
        if n_keep < W + 1:
            dev_rows = torch.arange(W + 1, device = batch_logits.device)
            row_pages = ((kvp + dev_rows) // PAGE_SIZE)
            row_offs = ((kvp + dev_rows) % PAGE_SIZE)
            bt_cpu = seq.block_index_tensor[:, :max_pages_batch].cpu()
            phys = torch.tensor([int(bt_cpu[0, int(rp)]) for rp in row_pages.tolist()],
                                dtype = torch.long, device = batch_logits.device)
            for layer in self.cache.layers.values():
                if layer is None:
                    continue
                try:
                    k_cache, v_cache = layer.get_kv(cache_seqlens, block_index, -1)
                except Exception:
                    continue
                if k_cache is None:
                    continue
                k_rows = k_cache[phys, row_offs]
                v_rows = v_cache[phys, row_offs]
                k_cache[phys[:n_keep], row_offs[:n_keep]] = k_rows[keep]
                v_cache[phys[:n_keep], row_offs[:n_keep]] = v_rows[keep]
            # page trim: release (W+1-n_keep) rows walking BACKWARD from the staged end
            end = kvp + W + 1
            r = (W + 1) - n_keep
            while r > 0:
                page = seq.allocated_pages[(end - 1) // PAGE_SIZE]
                rp = min(page.kv_position, r)
                assert rp > 0, "page trim underrun"
                page.kv_position -= rp
                end -= rp
                r -= rp

        # ---- GDN commit: direct row copy from retained bf16 node states + conv ring ----
        if not tree_out and batch_states is not None:
            # NOGDN probe: the normal sequential path advanced state through the whole
            # window; rewind like reject_remainder does (staged-history replay restores it)
            batch_states[0].rewind((W + 1) - n_keep)
        if _dbg:
            from ..util import cuda_sync_active; cuda_sync_active()
            print(f"[tree] kvcompact done {time.time()-_t0:.2f}s", flush = True)
        node_states_all = tree_out.get("node_states") or {}
        from ..modules.gated_delta_net_fn.gated_delta_rule import gated_delta_rule_tree
        chain_par = torch.tensor([-1] + list(range(acc_nodes)), device = par.device)
        for li in list((tree_out.get("conv_y") or {}).keys()):
            dims = tree_out["dims"][li]
            Hk, Hv, kdim, vdim, Dk, Dv = dims
            slot = tree_out["slots"][li]
            rsl = batch_states[0].cache.get_recurrent_layer((li, 0)) if batch_states is not None else None
            conv_pool, rec_pool = rsl.get_state_tensors() if rsl is not None else (None, None)
            if rec_pool is not None:
                if li in node_states_all:
                    ns_row = node_states_all[li][int(keep[-1])].to(rec_pool.dtype)
                    for srow in tree_out.get("state_rows", {}).get(li, [slot]):
                        rec_pool[srow, 0] = ns_row
                else:
                    # fallback: exact chain-walk recompute over the accepted path
                    beta = tree_out["beta"][li]; g = tree_out["g"][li]
                    conv_y = tree_out["conv_y"][li]
                    kdev = keep.to(conv_y.device)
                    base = rec_pool[slot, 0:1]
                    _, ns = gated_delta_rule_tree(
                        conv_y[:, kdev], beta[:, kdev], g[:, kdev], base, chain_par, Hk, Hv, kdim, vdim, Dk, Dv,
                    )
                    rec_pool[slot, 0] = ns[-1, 0]
            if conv_pool is not None:
                K = tree_out["tails"][li].shape[-1]
                conv_pool[slot, :, :K] = tree_out["tails"][li][int(keep[-1]), 0]

        # ---- job bookkeeping ----
        if _dbg:
            from ..util import cuda_sync_active; cuda_sync_active()
            print(f"[tree] gdn commit done {time.time()-_t0:.2f}s", flush = True)
        eos = False
        emitted = []
        # Emission follows the chain convention: logits row P predicts the token AT the child
        # node. For accepted path rows p1..pk the emitted rows are their PARENTS (path[:-1],
        # starting at row 0) and the bonus comes from the last accepted row path[-1]. Net:
        # emit every row of `path` in order, exactly like the chain's 0..k walk.
        for i, row in enumerate(path):
            token_logits = batch_logits[0:1, int(row):int(row) + 1, :]
            next_token, next_k_tokens, next_k_probs, next_prob = job.receive_logits(token_logits)
            eos, sampled_token, rq = job.receive_sample(
                token_logits, next_token, next_k_tokens, next_k_probs, next_prob, results,
            )
            emitted.append(int(sampled_token.item()) if torch.is_tensor(sampled_token) else -1)
            if i < acc_nodes:
                # accepted draft position (verified by the walk); keep filters/samplers in step
                job.accepted_draft_tokens += 1
                job.prepare_logit_mask()
                job.prepare_sampling_past_ids()
            if eos:
                job.rejected_draft_tokens += W - (i + 1)
                break
        if not eos:
            job.rejected_draft_tokens += W - acc_nodes
        if self.record_draft_stats:
            job.draft_stats.append((job.new_tokens, W, acc_nodes))

        # ---- draft state / cache alignment ----
        if self.draft_recurrent and job.recurrent_state:
            ds = job.recurrent_state.draft
            ds.rewind(max(0, ds.last_history - acc_nodes))
        if self.dflash_draft and p_export_states is not None:
            hid = p_export_states[-1] if not isinstance(p_export_states, list) else p_export_states
            def _gather(h):
                if torch.is_tensor(h) and h.dim() == 3:
                    return h[:, keep.to(h.device)]
                return h
            if isinstance(hid, list):
                hid = [_gather(h) for h in hid]
            else:
                hid = _gather(hid)
            self.draft_model.update_kv_from_target(
                target_hidden = hid,
                cache = self.draft_cache,
                lengths = [n_keep],
                params = {
                    "block_table": block_index,
                    "cache_seqlens": p_cache_seqlens,
                    "recurrent_states": [job.recurrent_state.draft] if self.draft_recurrent else None,
                },
            )
        if eos:
            completed = [job]
            job.deallocate_pages()
            self.active_jobs.remove(job)
        if _dbg:
            from ..util import cuda_sync_active; cuda_sync_active()
            print(f"[tree] round done {time.time()-_t0:.2f}s emitted={emitted}", flush = True)
        self.model.prefetch_tokens([])
