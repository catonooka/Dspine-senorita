"""
DSpine drafter (Tencent DSpine, arXiv 2609.36173) for exllamav3.

A block-causal parallel drafter built on the DFlash encoder machinery: the fc/hidden_norm
context projection, target-layer state export and draft-owned context K/V writes all come
from architecture/dflash.py. DSpine adds:

  * a selected-prefix code bridge: one write after every draft layer that codes the
    residual stream, injects the predecessor position's gated message and writes the
    result back into the residual stream (carrying causal dependencies through depth),
  * a serial refinement tail over the last layer's pre-write state, with vocabulary-wide
    tables TG [V, R] / TSV [V, 2R] built once from the target's LM head rows.

The forward and the tail mirror the DSpine MLX reference (dspine_mlx, Apache-2.0) rounding
point by rounding point; checkpoint tensors load under their native
``selected_prefix_bridge.`` names. Checkpoints: tencent/Qwen3.8-27B-DSpine-b16 and family
(architectures: DFlashDraftModel + dflash_config.selected_prefix_code_bridge).
"""
from __future__ import annotations

import math

import torch
from typing_extensions import override

from ..modules.transformer import TransformerBlock
from .dflash import DFlashConfig, DFlashModel

_F32_TINY = 1.1754943508222875e-38  # torch.finfo(torch.float32).tiny


class DSpineBridge:
    """Torch port of the DSpine code bridge. Tensors load at Model.load() time."""

    def __init__(self, config: "DSpineConfig"):
        self.config = config
        self.device = None
        self.code_size = config.bridge_code_size
        self.rank = config.bridge_rank
        self.eps = config.bridge_eps
        self.num_layers = config.num_hidden_layers
        self.sqrt_c = math.sqrt(self.code_size)
        self.read_weight = None            # [C, H]      last layer
        self.early_read_weights = None     # [L-1, C, H] layers 0..L-2
        self.feature_write_weights = None  # [L, H, R]
        self.codebook = None               # [V, C]
        self.packed_weight = None          # [C, 3R] = [gate.W; predecessor_gate.W; prev.W]^T
        self.gate_bias = None              # [R]
        self.norm_weight = None            # final norm weight [H], for the tail
        self._norm_weights = {}            # next_norm key -> bf16 weight tensor
        # Optional CUDA-graph capture of the per-layer write math (static B, T)
        import os
        self._write_fn = self._write_math
        if os.environ.get("DSPINE_COMPILE"):
            self._write_fn = torch.compile(self._write_math, dynamic = False)

    def load(self, device):
        stc = self.config.stc
        def g(name):
            return stc.get_tensor(name, device, allow_bf16 = True, no_defer = True)

        self.device = device
        self.read_weight = g("selected_prefix_bridge.read_weight")
        self.early_read_weights = g("selected_prefix_bridge.early_read_weights")
        self.feature_write_weights = g("selected_prefix_bridge.feature_write_weights")
        self.codebook = g("selected_prefix_bridge.codebook")
        gate_w = g("selected_prefix_bridge.gate.weight")
        self.gate_bias = g("selected_prefix_bridge.gate.bias").float()
        pred_w = g("selected_prefix_bridge.predecessor_gate.weight")
        prev_w = g("selected_prefix_bridge.prev.weight")
        self.packed_weight = torch.cat([gate_w, pred_w, prev_w], dim = 0).transpose(0, 1).contiguous()
        self.norm_weight = g("norm.weight")
        self._norm_weights["norm.weight"] = self.norm_weight
        for idx in range(self.config.num_hidden_layers):
            self._norm_weights[f"layers.{idx}.input_layernorm.weight"] = \
                g(f"layers.{idx}.input_layernorm.weight")

    def next_norm_weight(self, norm) -> torch.Tensor:
        for cand in (getattr(norm, "weight", None), self._norm_weights.get(norm.key + ".weight")):
            if cand is not None and torch.is_tensor(cand):
                return cand
        raise KeyError(f"[DSpine] no weight tensor for norm key {norm.key}")

    def layer_read_weight(self, layer: int) -> torch.Tensor:
        if layer == self.num_layers - 1:
            return self.read_weight
        return self.early_read_weights[layer]

    def _write_math(
        self,
        out,               # [B, T, H] half residual stream (post-MLP sum)
        anchor_codes,      # [B, 1, C] bf16, from the vocab codebook
        read_w,            # [C, H] this layer's read weight
        packed_w,          # [C, 3R]
        feat_write_w,      # [H, R] this layer's write weight
        gate_bias,         # [R] fp32
        next_norm_w,       # [H] bf16
        next_norm_eps,     # float
    ):
        """Compiled core (static B, T): bridge write + fused next-norm. MLX rounding points."""
        B, T, H = out.shape
        R = gate_bias.shape[-1]
        eps = self.eps
        sqrt_c = self.sqrt_c

        f = out.float()
        rms = torch.sqrt((f * f).mean(dim = -1).clamp_min(_F32_TINY))          # [B, T]
        normalized = (f / (rms[..., None] + eps)).to(torch.bfloat16)

        projected = torch.matmul(normalized, read_w.transpose(0, 1)).float()
        proj_norm = torch.sqrt((projected * projected).sum(dim = -1, keepdim = True))
        codes = (sqrt_c * projected / (proj_norm + eps)).to(torch.bfloat16)
        codes = torch.cat([anchor_codes, codes[:, 1:, :]], dim = 1)

        packed = torch.matmul(codes, packed_w)                                 # [B, T, 3R] bf16
        q, s, v = torch.split(packed.float(), [R, R, R], dim = -1)
        gate = torch.sigmoid(q[:, 1:, :] + s[:, :-1, :] + gate_bias)
        message = torch.cat(
            [torch.zeros_like(q[:, :1, :]), gate * v[:, :-1, :]], dim = 1
        ).to(torch.bfloat16)
        delta = torch.matmul(message, feat_write_w.transpose(0, 1)).float()

        not_anchor = (torch.arange(T, device = out.device) > 0).float().view(1, T, 1)
        out_b = (f + (rms[..., None] * delta) * not_anchor).to(torch.bfloat16)

        x = out_b.float()
        y = (x * torch.rsqrt(x.square().mean(dim = -1, keepdim = True) + next_norm_eps)).to(torch.bfloat16)
        normed = (y.float() * next_norm_w.float()).to(torch.bfloat16)
        # MLX stashes the PRE-write residual (bf16-rounded input) as raw
        raw_flat = out.to(torch.bfloat16).reshape(B * T, H)
        return normed, out_b.to(torch.half), rms.reshape(B * T), packed.reshape(B * T, packed.shape[-1]), raw_flat

    def write(
        self,
        out: torch.Tensor,           # [B, T, H] residual stream (post-MLP sum)
        anchor_ids: torch.Tensor,    # [B] anchor token ids (block position 0)
        layer: int,
        next_norm,
        stash: dict | None,
    ):
        anchor_codes = (self.sqrt_c * self.codebook[anchor_ids].float()).to(torch.bfloat16)[:, None, :]
        normed, residual, rms_flat, packed_flat, raw_flat = self._write_fn(
            out,
            anchor_codes,
            self.layer_read_weight(layer),
            self.packed_weight,
            self.feature_write_weights[layer],
            self.gate_bias,
            self.next_norm_weight(next_norm),
            next_norm.rms_norm_eps,
        )
        if stash is not None:
            # MLX stashes the bf16 residual; raw arrives bf16 from the compiled core
            stash["raw"] = raw_flat
            stash["rms"] = rms_flat
            stash["packed"] = packed_flat
        return normed, residual


class DSpineConfig(DFlashConfig):
    arch_string = "DFlashDraftModel"
    # DSpine's target_layer_ids (MLX reference) hook the OUTPUT of layer j. exllamav3's
    # exporter follows the HF hidden_states convention, so the equivalent export index
    # for the output of layer j is j + 1: tap_shift = 1 (same empirical fix as the
    # original DFlash release, see dflash.py)
    tap_shift = 1

    def __init__(
        self,
        directory: str,
        model_classes: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            directory,
            model_classes or {"text": DSpineModel},
            **kwargs
        )
        d = self.config_dict.get("dflash_config") or {}
        br = d.get("selected_prefix_code_bridge") or {}
        assert br, \
            "DSpineConfig: config.json has no dflash_config.selected_prefix_code_bridge"
        self.bridge_code_size = int(br["code_size"])
        self.bridge_rank = int(br["rank"])
        self.bridge_eps = float(br["eps"])

        sec = self.config_dict.get("code_bridge_second_pass") or {}
        refine = d.get("selected_prefix_last_write_refine") or {}
        widths = {int(x) for x in (sec.get("head_topk", 0), refine.get("head_topk", 0)) if x}
        assert len(widths) == 1, f"DSpine refinement widths disagree: {widths}"
        self.head_topk = widths.pop()
        assert self.head_topk & (self.head_topk - 1) == 0 and 2 <= self.head_topk <= 64

        fl = d.get("selected_prefix_feature_layers") or []
        feature_layers = [int(x) - 1 for x in fl]
        assert feature_layers == list(range(self.num_hidden_layers)), \
            "DSpine requires a bridge write after every draft layer"

        # DSpine blocks are causal (draft_block_causal); DFlashConfig may have read a
        # top-level is_causal: false that would disable the causal block mask
        self.is_causal = bool(d.get("draft_block_causal", True))
        self.dspine = True


class DSpineModel(DFlashModel):
    config_class = DSpineConfig

    _bridge_keys = (
        "read_weight", "early_read_weights", "feature_write_weights", "codebook",
        "gate.weight", "gate.bias", "predecessor_gate.weight", "prev.weight",
    )

    def __init__(self, config: DSpineConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.bridge = DSpineBridge(config)
        # Draft TransformerBlocks, in layer order (skip the input layer at index 0)
        first_block = self.first_block_idx
        self.draft_blocks = [
            m for m in self.modules[first_block:]
            if isinstance(m, TransformerBlock)
        ]
        assert len(self.draft_blocks) == config.num_hidden_layers, \
            f"[DSpine] expected {config.num_hidden_layers} blocks, got {len(self.draft_blocks)}"
        self._tg = None
        self._tsv = None
        # Staged draft-block upload state (elpis 0003)
        self._stage_key = None
        self._stage_mask_rows = None
        self._stage_blocks = {}
        import os
        self._score_fn = self._score_math
        if os.environ.get("DSPINE_COMPILE"):
            self._score_fn = torch.compile(self._score_math, dynamic = False)

    @staticmethod
    def _score_math(raw_nw, rms, q, head_rows, tsv_sid, tg_cand, gate_bias):
        """Compiled tail core (static B, n, K): base scores + predecessor message +
        serial score table; returns the argmax state table idx [B, n, K]."""
        R = q.shape[-1]
        base = torch.einsum("bih,bikh->bik", raw_nw, head_rows)          # [B, n, K]
        tvg, tvv = torch.split(tsv_sid, [R, R], dim = -1)
        m = torch.sigmoid(q[:, :, None, :] + tvg + gate_bias) * tvv      # [B, n, K, R]
        s = torch.matmul(m, tg_cand.transpose(-1, -2))                   # [B, n, K, K]
        score = base[:, :, None, :] + rms[:, :, None, None] * s
        return torch.argmax(score, dim = -1)                             # [B, n, K]

    @classmethod
    @override
    def get_additional_compiled_tensors(cls, config: DSpineConfig) -> dict:
        tensors = super().get_additional_compiled_tensors(config)
        stc = config.stc
        for k in cls._bridge_keys:
            tensors.update(stc.list_tensors(prefix = "selected_prefix_bridge." + k))
        return tensors

    @override
    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)
        self.bridge.load(self.modules[0].device)
        self._tg = None
        self._tsv = None

    def _staged_input_block(self, input_ids: torch.Tensor):
        """[B, 1] anchor ids -> [B, block, H] embedded block with a cached mask staging
        buffer (host embedding), uploaded async. Falls back to the full input layer when
        the embedding is device-resident."""
        embedding = self.attached_model().modules[0]
        emb_out = embedding.forward(input_ids, {})
        if emb_out.device.type != "cpu":
            # Device-resident embedding: full path (input layer) is already cheap
            return self.input_layer.forward(input_ids, {})
        bsz = input_ids.shape[0]
        block_len = self.config.block_size
        key = (id(embedding), getattr(getattr(embedding, "weight", None), "data_ptr", lambda: 0)())
        if self._stage_key != key:
            mask_ids = torch.full((1, block_len - 1), self.config.mask_token_id, dtype = torch.long)
            mask_rows = embedding.forward(mask_ids, {})
            if getattr(self.input_layer, "input_embedding_scale", 1.0) != 1.0:
                mask_rows = mask_rows * self.input_layer.input_embedding_scale
            self._stage_key = key
            self._stage_mask_rows = mask_rows
            self._stage_blocks = {}
        block = self._stage_blocks.get(bsz)
        if block is None:
            block = torch.empty(
                (bsz, block_len, self._stage_mask_rows.shape[-1]),
                dtype = self._stage_mask_rows.dtype,
                pin_memory = True,
            )
            block[:, 1:].copy_(self._stage_mask_rows.expand(bsz, -1, -1))
            self._stage_blocks[bsz] = block
        anchor_rows = emb_out
        if getattr(self.input_layer, "input_embedding_scale", 1.0) != 1.0:
            anchor_rows = anchor_rows * self.input_layer.input_embedding_scale
        block[:, :1].copy_(anchor_rows)
        return block.to(self.modules[0].device, non_blocking = True)

    @override
    def forward(self, input_ids: torch.Tensor, params: dict | None = None):
        """Block forward with bridge writes. Mirrors DSpineDraftModel.__call__ (MLX):
        layer input arrives pre-normalized (layer 0 from its own input_layernorm, later
        layers from the bridge's fused write), the residual stream is the bridge's
        pre-norm ``out``."""
        if params is None:
            params = {}
        x = self.prepare_inputs(input_ids, params)
        for m in self._get_prefetch_layers:
            m.prefetch(x, params)

        # Staged draft-block upload (elpis 0003-draft-graph technique): the target
        # embedding is host-resident; embed the constant mask rows ONCE into a pinned
        # buffer and upload only the anchor row per round, instead of embedding all 16
        # rows on the host every round.
        if input_ids.shape[1] == 1 and not self.loaded_tp:
            x = self._staged_input_block(input_ids)
        else:
            x = self.input_layer.forward(x, params)  # embed + mask append [B, T, H]
        # The input layer runs the TARGET's embedding, which may be pinned to host memory
        # and return a CPU tensor; forward_ls would hand x to the first block's
        # prepare_for_device, which my interleaved bridge loop bypasses
        x = self.draft_blocks[0].prepare_for_device(x, params)
        anchor_ids = input_ids[:, 0].to(x.device)
        # Block stream runs in half (ext kernels accept half/float only); the bridge's
        # internal math keeps the MLX bf16 rounding points
        residual = x.to(torch.half)
        x_normed = self.draft_blocks[0].attn_norm.forward(x, params, out_dtype = torch.half)

        last = len(self.draft_blocks) - 1
        stash_out = None
        for i, block in enumerate(self.draft_blocks):
            y = block.attn.forward(
                block.attn.prepare_for_device(x_normed.to(torch.half), params), params
            )
            h = residual + y.to(residual.dtype)
            m = block.mlp.forward(
                block.mlp_norm.forward(h, params, out_dtype = torch.half), params
            )
            out = h + m.to(h.dtype)

            next_norm = self.draft_blocks[i + 1].attn_norm if i < last else self.modules[-1]
            stash = {} if i == last else None
            normed, residual = self.bridge.write(out, anchor_ids, i, next_norm, stash)
            x_normed = normed
            if stash is not None:
                stash_out = stash

        params["dspine_stash"] = stash_out
        params["dspine_anchor_ids"] = anchor_ids
        return x_normed

    # ------------------------------------------------------------------ tail

    def _ensure_tables(self, target):
        """TG [V, R] and TSV [V, 2R], BF16, per the MLX build_serial_tables."""
        import os
        dbg = os.environ.get("DSPINE_DEBUG")
        if self._tg is not None:
            return
        lm = target.modules[target.logit_layer_idx]
        device = lm.device
        # The bridge's codebook defines the reachable vocab; the target head may be padded
        V = min(target.config.vocab_size, self.bridge.codebook.shape[0])
        R = self.bridge.rank
        norm_w = self.bridge.norm_weight.float().to(device)          # [H]
        write_w = self.bridge.feature_write_weights[-1].to(device)   # [H, R]
        pv_w = self.bridge.packed_weight[:, R:].to(device)           # [C, 2R]
        sqrt_c = self.bridge.sqrt_c
        if isinstance(sqrt_c, torch.Tensor): sqrt_c = sqrt_c.to(device)

        tg_parts, tsv_parts = [], []
        try:
            free_b, _ = torch.cuda.mem_get_info(device)
            chunk = 2048 if free_b < 2 * 1024 ** 3 else 8192
        except Exception:
            chunk = 8192
        with torch.inference_mode():
            for start in range(0, V, chunk):
                end = min(start + chunk, V)
                if dbg: print(f"[T] chunk {start}-{end}", flush = True)
                ids = torch.arange(start, end, device = device)
                rows = _head_rows(lm, ids)                             # [c, H] bf16
                if dbg: torch.cuda.synchronize(); print(f"[T] rows {tuple(rows.shape)}", flush = True)
                tg_parts.append(
                    torch.matmul((rows.float() * norm_w).to(torch.bfloat16), write_w)
                )
                tsv_parts.append(
                    torch.matmul(
                        (sqrt_c * self.bridge.codebook[start:end].to(device).float()).to(torch.bfloat16),
                        pv_w,
                    )
                )
        self._tg = torch.cat(tg_parts).contiguous()     # [V, R]
        self._tsv = torch.cat(tsv_parts).contiguous()   # [V, 2R]

    @override
    def sample_from_state(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        """Serial refinement over the final layer's pre-write state. Greedy chain over
        the head's top-K candidates; returns [B, T] with the anchor at column 0."""
        target = self.attached_model()
        self._ensure_tables(target)

        stash = params.get("dspine_stash")
        assert stash is not None, "DSpine forward must run before sample_from_state"

        # Eager part: candidates through the target head (custom EXL3 kernels stay
        # outside the graph; capture-unsafe)
        cand, head_rows = self._tail_candidates(state, params, target)

        # CUDA-graph path (elpis host-overlap philosophy): the score+chain section is
        # pure static-shape tensor math — capture once, replay per round.
        graph = getattr(self, "_tail_graph", None)
        import os as _os
        if graph is not None:
            self._g_cand.copy_(cand)
            self._g_head_rows.copy_(head_rows)
            self._g_raw.copy_(stash["raw"])
            self._g_rms.copy_(stash["rms"])
            self._g_packed.copy_(stash["packed"])
            self._g_anchor.copy_(params["dspine_anchor_ids"])
            graph.replay()
            if _os.environ.get("DSPINE_LATTICE"):
                self._lattice_capture(self._g_cand, self._g_head_rows,
                                      {"raw": self._g_raw, "rms": self._g_rms, "packed": self._g_packed},
                                      self._g_anchor)
            return self._g_out.clone()
        import os
        if os.environ.get("DSPINE_GRAPH") and torch.cuda.device_count() == 1:
            self._tail_graph_capture(cand, head_rows, stash, params)
        out = self._tail_rest(cand, head_rows, stash, params["dspine_anchor_ids"])
        if os.environ.get("DSPINE_LATTICE"):
            self._lattice_capture(cand, head_rows, stash, params["dspine_anchor_ids"])
        return out

    def _tail_graph_capture(self, cand, head_rows, stash, params):
        """Build static buffers, warm up, then capture the score+chain section."""
        import torch.cuda.graphs as _g
        dev = cand.device
        self._g_cand = cand.clone()
        self._g_head_rows = head_rows.to(dev).clone()
        self._g_raw = stash["raw"].to(dev).clone()
        self._g_rms = stash["rms"].to(dev).clone()
        self._g_packed = stash["packed"].to(dev).clone()
        self._g_anchor = params["dspine_anchor_ids"].to(dev).clone()
        for _ in range(2):
            self._g_out = self._tail_rest(self._g_cand, self._g_head_rows,
                                          {"raw": self._g_raw, "rms": self._g_rms, "packed": self._g_packed},
                                          self._g_anchor)
        torch.cuda.synchronize()
        graph = _g.CUDAGraph()
        with torch.cuda.device(dev), _g.graph(graph):
            self._g_out = self._tail_rest(self._g_cand, self._g_head_rows,
                                          {"raw": self._g_raw, "rms": self._g_rms, "packed": self._g_packed},
                                          self._g_anchor)
        self._tail_graph = graph

    def _lattice_capture(self, cand, head_rows, stash, anchor_ids):
        """Research hook (DSPINE_LATTICE=1): rebuild the [n, K, K] edge score tensor
        (log-softmax over candidate k given predecessor j) for offline tree replay.
        Preprocessing mirrors _tail_rest; runs eagerly on the round's live tensors."""
        device = cand.device
        B, n, K = cand.shape
        R = self.bridge.rank
        raw = stash["raw"].to(device).view(B, n + 1, -1)[:, 1:].float()
        rms_ = stash["rms"].to(device).view(B, n + 1)[:, 1:].float()
        q = stash["packed"].to(device).view(B, n + 1, 3 * R)[:, 1:, :R].float()
        anchor_ids = anchor_ids.to(device)
        norm_w = self.bridge.norm_weight.float().to(device)
        gate_bias = self.bridge.gate_bias
        if isinstance(gate_bias, torch.Tensor): gate_bias = gate_bias.to(device)
        anchor_k = anchor_ids[:, None].expand(B, K)
        sid = torch.cat([anchor_k[:, None, :], cand[:, :-1, :]], dim = 1)
        tsv_sid = self._tsv[sid.reshape(-1)].view(B, n, K, 2 * R).float()
        tg_cand = self._tg[cand.reshape(-1)].view(B, n, K, R).float()
        base = torch.einsum("bih,bikh->bik", raw * norm_w, head_rows.to(device).float())
        tvg, tvv = torch.split(tsv_sid, [R, R], dim = -1)
        m = torch.sigmoid(q[:, :, None, :] + tvg + gate_bias) * tvv
        s = torch.matmul(m, tg_cand.transpose(-1, -2))
        score = base[:, :, None, :] + rms_[:, :, None, None] * s      # [B, n, K(pred j), K(cand k)]
        self._lattice_out = (
            cand[0].detach().cpu(),                                  # [n, K] candidate ids
            torch.log_softmax(score[0].float(), dim = -1).cpu(),      # [n, K, K] edge log-probs
        )

    def sample_tree_from_state(self, state: torch.Tensor, params: dict, cap: int = 31):
        """TREE draft path: build a best-first tree over the lattice (log-softmax edge
        scores over the K candidates of each predecessor row), topological row order.
        Returns (tokens [B, 1+cap], parent [B, 1+cap]) with row 0 = anchor, parent[0] = -1.
        Eager math only — never captured, never on the champion chain path."""
        import heapq
        target = self.attached_model()
        self._ensure_tables(target)
        stash = params.get("dspine_stash")
        assert stash is not None
        cand, head_rows = self._tail_candidates(state, params, target)

        device = cand.device
        B, n, K = cand.shape
        R = self.bridge.rank
        raw = stash["raw"].to(device).view(B, n + 1, -1)[:, 1:].float()
        rms = stash["rms"].to(device).view(B, n + 1)[:, 1:].float()
        q = stash["packed"].to(device).view(B, n + 1, 3 * R)[:, 1:, :R].float()
        anchor_ids = params["dspine_anchor_ids"].to(device)
        norm_w = self.bridge.norm_weight.float().to(device)
        gate_bias = self.bridge.gate_bias
        if isinstance(gate_bias, torch.Tensor): gate_bias = gate_bias.to(device)
        anchor_k = anchor_ids[:, None].expand(B, K)
        sid = torch.cat([anchor_k[:, None, :], cand[:, :-1, :]], dim = 1)
        tsv_sid = self._tsv[sid.reshape(-1)].view(B, n, K, 2 * R).float()
        tg_cand = self._tg[cand.reshape(-1)].view(B, n, K, R).float()
        base = torch.einsum("bih,bikh->bik", raw * norm_w, head_rows.float())
        tvg, tvv = torch.split(tsv_sid, [R, R], dim = -1)
        m = torch.sigmoid(q[:, :, None, :] + tvg + gate_bias) * tvv
        s = torch.matmul(m, tg_cand.transpose(-1, -2))
        score = base[:, :, None, :] + rms[:, :, None, None] * s       # [B, n, K, K]
        lps = torch.log_softmax(score.float(), dim = -1).cpu()        # [B, n, K, K]

        out_t = torch.full((B, 1 + cap), -1, dtype = torch.long)
        out_p = torch.full((B, 1 + cap), -1, dtype = torch.long)
        for b in range(B):
            cr = cand[b].cpu()
            lr = lps[b]
            toks = [int(anchor_ids[b])]
            pars = [-1]
            slots = [-1]
            heap = [(-float(lr[0, 0, k]), 0, k, 0) for k in range(K)]
            heapq.heapify(heap)
            while heap and len(toks) <= cap:
                neglp, slot, k, par = heapq.heappop(heap)
                row = len(toks)
                toks.append(int(cr[slot, k])); pars.append(par); slots.append(slot)
                if slot + 1 < n:
                    for k2 in range(K):
                        heapq.heappush(heap, (neglp - float(lr[slot + 1, k, k2]), slot + 1, k2, row))
            w = len(toks)
            out_t[b, :w] = torch.tensor(toks)
            out_p[b, :w] = torch.tensor(pars)
            # slots per row for the verifier (depth = slot+1)
            out_p[b][0] = -1
        depths = torch.tensor([0] + [s + 1 for s in slots[1:]]) if len(slots) > 1 else torch.zeros(1, dtype = torch.long)
        return out_t, out_p, depths

    def _tail_candidates(self, state: torch.Tensor, params: dict, target):
        """top-K candidates + dequantized head rows through the target's quantized head."""
        B, T, H = state.shape
        n = T - 1
        K = self.config.head_topk
        V = min(target.config.vocab_size, self.bridge.codebook.shape[0])
        lm = target.modules[target.logit_layer_idx]
        rows = state[:, 1:, :]
        logits = lm.forward(lm.prepare_for_device(rows.half(), params), params)
        logits = logits[..., :V]
        cand = torch.topk(logits.float(), K, dim = -1).indices
        cand, _ = torch.sort(cand, dim = -1)
        head_rows = _head_rows(lm, cand.reshape(-1)).float().view(B, n, K, H)
        return cand, head_rows

    def _tail_rest(self, cand: torch.Tensor, head_rows: torch.Tensor, stash: dict, anchor_ids: torch.Tensor) -> torch.Tensor:
        B, n, K = cand.shape
        R = self.bridge.rank
        device = cand.device
        T = n + 1

        raw = stash["raw"].to(device).view(B, T, -1)[:, 1:].float()       # [B, n, H]
        rms = stash["rms"].to(device).view(B, T)[:, 1:].float()          # [B, n]
        q = stash["packed"].to(device).view(B, T, 3 * R)[:, 1:, :R].float()  # [B, n, R]
        anchor_ids = anchor_ids.to(device)

        norm_w = self.bridge.norm_weight.float().to(device)
        gate_bias = self.bridge.gate_bias
        if isinstance(gate_bias, torch.Tensor): gate_bias = gate_bias.to(device)

        # Predecessor tables: anchor for position 0, candidate i-1 afterwards
        anchor_k = anchor_ids[:, None].expand(B, K)                      # [B, K]
        sid = torch.cat([anchor_k[:, None, :], cand[:, :-1, :]], dim = 1)  # [B, n, K]
        tsv_sid = self._tsv[sid.reshape(-1)].view(B, n, K, 2 * R).float()
        tg_cand = self._tg[cand.reshape(-1)].view(B, n, K, R).float()
        idx = self._score_fn(raw * norm_w, rms, q, head_rows, tsv_sid, tg_cand, gate_bias)

        # Walk the chain: start from the argmax state at row 0 (reference _chain),
        # then state j at row i picks candidate idx[b, i, j]
        toks = torch.empty((B, n), dtype = torch.long, device = device)
        ar = torch.arange(B, device = device)
        j = idx[ar, 0, 0]
        for i in range(n):
            toks[:, i] = torch.gather(cand[ar, i], 1, j[:, None]).squeeze(1)
            if i + 1 < n:
                j = torch.gather(idx[ar, i + 1], 1, j[:, None]).squeeze(1)
        return torch.cat([anchor_ids[:, None], toks], dim = 1)           # [B, T]


def _head_rows(lm, ids: torch.Tensor) -> torch.Tensor:
    """Dequantized head rows for token ids: [*, H] for ids [*]. reconstruct_columns
    fills the weight's columns ([H, n]); head row i is column i of the [V, H] head."""
    inner = lm.inner if hasattr(lm, "inner") else lm
    flat = ids.reshape(-1)
    H = inner.in_features if hasattr(inner, "in_features") else lm.in_features
    out = torch.empty((H, flat.numel()), dtype = torch.half, device = lm.device)
    inner.reconstruct_columns(flat, out)
    rows = out.transpose(0, 1).to(torch.bfloat16)
    return rows.view(*ids.shape, -1)
