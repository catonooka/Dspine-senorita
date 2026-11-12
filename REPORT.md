# DSpine-Senorita — Deep Report

**Campaign:** Qwen3.8-27B EXL3 4.0bpw-h8 on 1× RTX 3090 (production single-card rule) —
maximize creative-prose decode speed with parallel drafters (DSpine vs DFlash2),
dynamic verification, and drafter re-quantization.

**Period:** 2026-11-10 → 2026-11-12. All numbers are measured, same-day A/B, greedy,
single stream, GPU locked (350 W / 1770 MHz / mem 9501) unless labeled projection.

---

## 1. TL;DR

| Lane | Config | t/s (400-tok story) | Committed/round |
|---|---|---:|---:|
| **Champion (prose)** | **DFlash2-EXL3-5.0bpw, window 7, dynamic c0.25** | **64.7 think / 67.0 no-think** | 2.67 / 2.55 |
| DFlash2 BF16, dyn c0.25 | same, BF16 drafter | 58.0 / 60.0 | 3.07 / 2.46 |
| DSpine BF16, dyn15 c0.25 | full-block + dynamic verify | 60.0 / 43.2 | 2.78 / 2.08 |
| DSpine EXL3, dyn15 c0.25 | 5bpw drafter | 57.5 / 44.3 | 2.60 / 2.06 |
| Dense floor (no drafter) | — | 42.4 | 1.0 |

220-token bench scale reads higher (shorter runs, warm acceptance): DFlash2 BF16
static ndt7 hit **75.1–83.8 t/s think / 69.0 no-think**; DSpine static **72.3 t/s think**.
Never compare across run lengths — within-sweep only.

**Verdict:** DFlash2 with an EXL3-quantized drafter + confidence-truncated verification
wins BOTH prose lanes. DSpine keeps the code lane (8.16 committed/round vs 3.49 chain
in tree mode; ~72% acceptance on code vs 14–30% on creative prose).

---

## 2. DSpine on paper vs on our rig

**Paper** (arXiv 2609.36173, tencent/DSpine, official Qwen3.8-27B-DSpine-b16):

- 5-layer **parallel** drafter: the whole 16-token block is drafted in one forward
  pass. Intra-block causal dependencies are modeled through *depth* — each draft layer
  passes the predecessor position's predicted feature to its neighbor (Adjacent
  Causal Injection). No sequential drafting steps.
- Block 16 vs DFlash2's 8 — double the speculative window per pass.
- Acceptance length on 27B (SGLang, greedy): MT-Bench 4.48 vs DFlash2's 4.11;
  agent benchmarks +13–33% DSpine everywhere.
- End-to-end throughput 2.71–3.84× AR vs DFlash2's 2.33–3.17×.
- The headline wall-clock trick: **draft always full 16, verify width DYNAMIC (4–16),
  allocated by drafter confidence** (DSpark-style). Never pay 16-row verify on a
  low-confidence round.

**What we measured** (exllamav3 port, creative story, greedy):

- Acceptance: DSpine 14.0% no-think / 19.2–28.2% think vs DFlash2 22.3% / 26.0–30.0%
  — the acceptance ordering is **flipped** vs the paper's MT-Bench table.
- After running the paper's exact recipe (full-16 draft + dynamic verify):
  DSpine 60.0 t/s think vs DFlash2 64.7–67.0. Still behind.

**Why the paper advantage does not transfer:**

1. **Workload entropy.** MT-Bench answers and agent trajectories are low-entropy
   continuations; creative fiction is the hardest lane. DSpine's 5-layer depth
   conditioning is weakest exactly where next-token entropy is highest — each draft
   position only sees ~5 "looks" at predecessors. DFlash2's selector walk over the
   block holds up better on high-entropy tails.
2. **Verify-dominated rounds.** DSpine's wider block means wider verify passes; the
   acceptance gain per drafted token is spent on verify rows. Static ndt15 and ndt7
   both landed at exactly 72.3 t/s — the wider block's extra committed tokens were
   eaten by verify cost. Dynamic verify only partially recovers this.
3. **Drafter quant asymmetry.** EXL3-5bpw cost DSpine acceptance (2.78 → 2.60
   committed) but barely touched DFlash2 — so the quant's round-cost saving pays
   only for DFlash2 (+11.6/11.7%).
4. **BF16 draft cost was never DSpine's wall** on our rig (draft ~11 ms of a
   ~45 ms round); the wall is acceptance × verify width.

---

## 3. Campaign ladder (all measured)

| Step | Result |
|---|---|
| Dense floor, window sweep | 42.4 t/s floor; verify pass = 42 ms / 703 launches |
| DSpine port (elpis-style attach) | 44.76 no-think / 72.3 think (220-tok) |
| Round anatomy | prose round 44.6 ms = draft ~11 + verify ~23 + overhead; code round 69.2 ms |
| KV 8-bit | output-safe (619-tok bit-identical); 128K ctx 169.9 t/s prefill-side |
| Tree verification port | 6 bugs fixed → bit-comparable to chain within engine noise (prefix 4–31) |
| Tree wall-clock | 11.9 t/s prose / 38.2 code — launch-bound (~2500 launches, ~240 ms rounds); loses wall-clock despite 2.3× committed |
| DFlash2 prose A/B | 69.0 no-think / 75.1–83.8 think (220-tok) vs DSpine 47.8 / 55.6 |
| Dynamic verify sweep | engine calibrator; DSpine best c0.25; default 0.4 worst (U-shaped) |
| EXL3 drafters (ours, 5.0bpw-h8) | DFlash2 64.7/67.0 @400-tok = champion; DSpine flat |

## 4. Tree verification (why it's parked)

Ported DSpine's lattice tree into exllamav3 (16-row tree verify). Six root-cause bugs
fixed (conv ring head-vs-tail rebase, GDN base-state row, `generator_stage_decode`
must take `ids=None`, missing `recurrent_history: True`, parent-row emission,
one-char `> 0` → `>= 0` path walk). Correctness proven: tree-vs-chain identical
prefix within the chain-vs-chain noise band (flash-attn summation order flips
near-ties across physical pages — byte-exactness is not achievable in-engine).

Wall-clock verdict: rounds are launch-bound (~240 ms). Code tree halves rounds
(25 vs 49, 8.16 committed vs 3.49) but still loses wall-clock. Next lever = single
Triton tree-delta kernel (est 60–80 ms rounds), then verify CUDA-graph. Parked.

## 5. Laws (do not re-test without a changed engine)

- Static window on a block drafter is always dominated by dynamic verify + full block.
- Confidence target is U-shaped: 0.25 (trust drafter longer) beat the 0.4 default for
  prose on both drafters' best lanes.
- Row-crop patches (rocm fork) are **moot** in dynamic mode: the calibrator needs
  every row's head logits to decide truncation.
- Drafter EXL3 quant changes only *which* drafts get accepted — the target still
  verifies everything, so outputs stay target-faithful (no quality gate needed).
- DSpine high-variance acceptance on creative prose (19–28% run-to-run on identical
  config) — single-run verdicts on it are noise.
- 400-tok runs read ~15% lower than 220-tok runs of the same config (acceptance
  regime shift as the story deepens). Compare within one run length only.

## 6. Serving the champion

```
CUDA_VISIBLE_DEVICES=0 ~/exl3-env/bin/python ~/Dspine-senorita/serve/serve_prose.py --port 8900
```

OpenAI-compatible: `POST /v1/chat/completions` (stream + non-stream), `GET /v1/models`.
Default lane = champion (DFlash2-EXL3, window 7, dynamic, c0.25). `--lane dspine`
available for comparison; `--no-dynamic` for static.

## 7. Next levers (unexercised)

1. Verify-pass CUDA graph (42 ms → est 25–30 ms) — lifts every lane ~15–20%.
2. Triton tree-delta kernel → tree mode viable (est 60–80 ms rounds).
3. Stacked: graph + dynamic + tree → think projected ~95–105 (projection, not measured).
4. 120 t/s creative prose needs ~4.8 committed/round — no current drafter predicts
   creative fiction at that rate; the honest ceiling without a better drafter is lower.

---

# SpineCamp: 120 prose / 200 code push (2026-11-12, DSpine only)

Goal: EXL3 4bpw target, single 3090, DSpine drafter only (no DFlash2), vision+thinking on.
Bench: 400 tok greedy, GPU1 locked 350W/1770/9501, DSpine-EXL3-5.0bpw-h8, window 15.

## Camp 1 — variant grid (spinecamp1.json)

| Variant | Code t/s | committed | Story t/s | committed |
|---|---:|---:|---:|---:|
| static15 | 121.2 | 5.48 | 50.1 | 2.30 |
| dyn c0.25 | **163.8** | 7.55 | 59.5 | 2.72 |
| static15 + hyb8 | 152.6 | 7.14 | **62.1** | 2.86 |
| static15 + hyb16 | 150.0 | 7.02 | 54.0 | 2.48 |
| dyn + hyb8 | 145.4 | 6.78 | 58.2 | 2.68 |

Laws: dynamic verify dominates code (+35% over static, 53 rounds); n-gram hybrid helps
prose only as STATIC+hyb8 (+4.4%); hybrid and dynamic stack NEGATIVELY (n-gram drafts
disagree with truncation policy, wasted verify width); hyb16 too wide for prose.

## Camp 2 — confidence + sparse head (spinecamp2.json)

| Variant | Code t/s | Story t/s |
|---|---:|---:|
| dyn c0.35 | 153.6 | — |
| dyn c0.25 + sparse32k | 147.1 | 52.4 (dyn) |
| dyn c0.35 + sparse32k | 121.6 | — |
| static15 + hyb8 + sparse32k | — | 62.4 |

Laws: sparse draft head REJECTED — costs code −10% (subset drops good draft tokens),
neutral on prose. Confidence: c0.25 stays optimal (higher targets truncate too eagerly).

## Camp 3 — fp16 draft head (Drafter-DSpine-EXL3-5.0bpw-h16, spinecamp3.json)

Code 165.3 (+0.9% ≈ noise, committed identical 7.55), story 56.0 (worse, 2.55 vs 2.86
committed). REJECTED — head precision is not DSpine's acceptance bottleneck. h8 stays.

## Camp 4 — noise band + verify CUDA graph (spinecamp4.json, vg_probe2.log)

3 reps × 400 tok, same process, GPU1 locked 350W/1770:
- code static15: 123.4 / 136.7 / 141.9 → mean 134.0
- code dyn c0.25: 111.7 / 143.4 / 138.0 → mean 131.0
- story static15+hyb8: 49.3 / 57.8 / 64.7 → mean 57.3

Static vs dynamic on code is a WASH once repeated (camp1's 163.8 = single-rep
trajectory luck). Spread source: kernel nondeterminism → near-tie flips → different
greedy trajectories → different acceptance profiles (committed 5.2-6.7 code, 2.3-3.0
story across reps of the SAME config).

**Verify-pass CUDA graph: conclusively NEGATIVE.** Implementation exists
(DSPINE_VERIFY_GRAPH=1 in generator.py: per-(rows,pages,ids-width) lazy capture with
static staging arenas; the arena tensors ARE refreshed in place each round, so the
design was sound). Capture dies with torch AcceleratorError inside the target forward
— an allocation/op not legal under stream capture (paged-cache growth during verify).
The apparent +10% in early A/Bs was acceptance-trajectory noise (graphs=0 at exit;
round_ms identical 45.5 vs 45.7). LAW: verify pass not graphable without moving cache
growth out of the captured region — days of surgery for ~10ms/round.

## Camp 5 — 6bpw drafter body (Drafter-DSpine-EXL3-6.0bpw-h8, spinecamp5.json)

code static15 142.2/134.3 (mean 138.3 vs 5bpw 134.0), code dyn 131.6 (wash), story 54.7
(wash). +0-3% at best, inside the noise band. VERDICT: keep Drafter-DSpine-EXL3-5.0bpw-h8
(smaller, equal); 6bpw retained as comparison artifact.

# Campaign close-out vs targets (single 3090, DSpine only, vision+thinking on)

| Lane | Target | Measured (mean, 400 tok ×3) | Best single | Status |
|---|---:|---:|---:|---|
| Code | 200 | 131-138 | 143.4 (187.4 lucky trajectory, 200-tok) | MISSED −30% |
| Prose story | 120 | 55-57 | 64.7 | MISSED −55% |

Physics: committed/round = 6.2 code / 2.7 story at 42-46 ms rounds. 200 t/s code needs
8.5 committed (drafter acceptance ceiling) or <30 ms rounds (verify graph dead — see
camp 4 law; only path = Triton tree-delta kernel, est days). 120 t/s prose needs 5+
committed — DSpine's high-entropy creative-text weakness (paper's own gains shrink there;
DFlash2 does 2.5/round on this bench — no drafter in reach of 5).

Levers tried and closed: dynamic verify (✓ code, wash on reps), n-gram hybrid (✓ prose
+5-8%, static+hyb8), sparse head (✗), h16 head (✗), 6bpw body (flat), verify CUDA
graph (✗ AcceleratorError), window sweeps, EXL3 drafter quant (✓ 5bpw-h8, serving).

Still open if we ever reopen: Triton tree-delta kernel (launch-bound 240ms tree rounds
→ 60-80ms), cache-growth-outside-capture to resurrect verify graph, recurrent
max_history>16 to unlock TREE_CAP>15.

---

# Why is DSpine slow here — phase anatomy + width-cost curve (2026-11-12, GPU1+GPU2)

## Phase anatomy (DSpine-EXL3-5.0bpw-h8, 400 tok, why_slow1.json)

| Config | t/s | round_ms | draft_ms | gen_ms | verify width used |
|---|---:|---:|---:|---:|---|
| story dyn c0.25 | 62.2 | 46.2 | 9.0 | 37.5 | **15.0 always (139/139 rounds full width)** |
| story static15 | 46.9 | 47.1 | 8.8 | 38.5 | 15.0 |
| code static15 | 110.5 | 50.3 | 10.5 | 40.5 | 15.0 |

## Target verify cost vs width (why_slow2.json, story, static widths)

| width | gen_ms | committed |
|---|---:|---:|
| w1 (2 rows) | 26.3 | 1.62 |
| w3 | 27.3 | 2.40 |
| w7 | 30.5 | 3.03 |
| w15 | 33.3 | 2.88 |
| DFlash2 w7 (ref) | 29.6 | 2.83 |

## DFlash2 reference (same harness, GPU2)

| Config | draft_ms | gen_ms | round_ms | proj t/s |
|---|---:|---:|---:|---:|
| df2 w7 static | **4.3** | 29.6 | 33.9 | 83.5 |
| df2 w7 dyn c0.25 | **4.4** | 27.3 | 31.7 | 95.9 |

## Diagnosis (four stacked reasons)

1. **Verify pass is ~75% FIXED cost.** 2 rows cost 26.3 ms, 16 rows cost 33.3 ms —
   marginal row ≈ 0.5 ms. The dense single-token floor is 23.6 ms, so nearly all of
   the verify pass is the per-round irreducible forward (≈703 kernel launches through
   62 layers + GDN state ops), not the drafted rows. Wider drafts are almost free;
   narrower drafts save almost nothing.
2. **Dynamic verify never truncates on our stack.** width histogram: 139/139 story
   rounds at full 15+1 rows even with c0.25 — DSpine per-row confidence rarely crosses
   the threshold, so the DSpark paper's mechanism doesn't engage. The earlier dyn-vs-static
   t/s differences were acceptance-trajectory luck, not saved verify work.
3. **DSpine's draft pass costs 2× DFlash2's** (8.7–10.5 ms vs 4.3–4.4 ms, both EXL3
   5bpw-h8). 5 ACI layers at width 16 with the 248k×1024 codebook vs DFlash2's blocks.
   That is an 11% round-time penalty on every round.
4. **Acceptance is the multiplier ceiling**: committed ≈ 2.2–3.0 on creative prose for
   BOTH drafters (DSpine 2.88, DFlash2 2.83/3.04 here). Combined with the 42 ms round
   (26 fixed verify + 9 draft + overhead) that pins prose at ~60–70 t/s regardless of
   width/confidence tuning.

## What would actually move it

- **Cut the verify fixed cost** (26 ms → 15–18 ms): hoist cache growth out of the
  captured region (camp-4 law: capture currently dies on in-forward allocation), or
  reduce launches per round. Projected: prose ≈ 90–95, code ≈ 140–150.
- **Halve draft cost**: extend DSPINE_GRAPH to the whole draft block (drafter is small;
  fewer capture hazards than the target). Projected: +10% round time.
- **Acceptance on prose**: no quant knob reached it (camps 1–5); needs a better
  drafter or tree verification (Triton tree-delta kernel, days of work).

# Blocker work log (2026-11-12, post-research)

## Upstream survey — master == v1.6.6 (bd1f519); dev = 27 commits ahead (4db6a855)
The API "graph commits" (GDN b/a GEMV single-pass d33d1869, batched spec-window
readback 288d0434, PR #489 replay) all predate v1.6.6 — already in our tree (squashed
clone hides SHAs; content verified by date). dev adds pipelined prefill, rolling
checkpoints, QSA/BlockSparse — nothing for verify-graph. **No ready-made verify-graph
runner exists upstream; the campaign's own surgery is the only path.**

## Blocker 1 — verify CUDA graph: 3 obstacles unpeeled (vgtrace1-3, GPU2)
1. GDN decode/verify already runs per-layer through the extension's OWN BC graph
   (BC_GatedDeltaNet(Split), graph.cu: cudaGraphExecKernelNodeSetParams + cudaGraphLaunch).
   torch.cuda.graph() capture nests over it → `operation not permitted when stream is
   capturing` at graph.cu:252. Routed around with `EXL3_GDN_FORCE_TORCH=1` (forces the
   plain-kernel torch path; +1-line condition in gated_delta_net.py).
2. Next blocker: multilinear `c_ptrs` builds a fresh CPU→GPU pointer table every call
   (multilinear.py:123, illegal pageable H2D in capture). Routed around with
   `EXL3_GRAPH_SAFE=1`: per-(m) persistent qkv/z/xh buffers + cached c_ptrs
   (gated_delta_net.py project_qkvz_sliced).
3. Next: capture now runs deeper but the stream gets invalidated mid-capture
   (cudaErrorStreamCaptureInvalidated at capture_end) — at least one more illegal op
   further in (conv/scan/export machinery), AND the aborted capture leaves engine state
   dirty enough that the following eager draft phase hits `A and B incompatible shapes`
   in the drafter's input projection. Status: **capture not yet green; two env-gated
   engine patches land the groundwork; remaining depth unknown.**

Law: the ~26 ms "fixed" verify cost is NOT all launches — the 48 GDN layers already
replay through per-layer BC graphs (48 launches). The ungraphed remainder is the
~650 launches of attention/MLP/norm/gemm across 64 layers; that is what whole-pass
capture would remove.

## Blocker 2 — block-width reps (bw_reps.json, GPU1, 3 reps each)
| width | story t/s | code t/s |
|---|---:|---:|
| 8 | **60.2** (58.6/67.8/54.1) | 127.1 |
| 11 | 58.8 | 111.9 |
| 15 | 54.8 | **143.9** (140.3/140.1/151.2) |

Prose wants width 8 (+10% mean over 15, bands overlap), code wants full 15. Per-lane
width is a serving decision (prose lane w8, code lane w15); not applied to the default
lane yet.

## Blocker 3 — closed (SPS non-lever at bs=1, see RESEARCH.md). Blocker 4 — trees deferred until fused verify exists.

## Blocker 1 — capture surgery log (2026-11-12 evening, GPU2)
- Architecture rewrite shipped: capture-FIRST (record from true pre-state, execute once via
  replay-once), export_states pool-stash for replay rounds, batch_ids copy_ into captured
  tensor, per-(slot,parity) graph key. Gate bug fixed (string-truthy "0").
- RESULT: capture SUCCEEDS (VGCAP), capture-round replay is BIT-EXACT vs eager
  (max_abs=0.0000, argmax 1.000). Post-capture corruption + fc shape death: GONE (COMPLETED_CLEAN).
- REMAINING: replays on LATER rounds diverge (max_abs 30-43, argmax 0.0). Parity key
  (2 graphs) did not fix. Root class: per-round host-side values baked into kernel launch
  args at record time (GDN position-derived decay/scan offsets, slots via _get_slot_tensor
  device-cache, possible KV write offsets). Next step = static-state capture (vLLM-style):
  dedicated graph slot rows, position-independent scan, tensor-driven offsets.
- Attention qkv c_ptrs capture-illegal copy fixed same as GDN (persistent bufs, EXL3_GRAPH_SAFE).
- Env matrix to reproduce: DSPINE_VERIFY_GRAPH=1 EXL3_GDN_FORCE_TORCH=1 EXL3_GRAPH_SAFE=1
  EXL3_BC_ATTN=0 (+DEBUG/TELLERY/SELFCHECK probes).

## Blocker 2 — clean draft/verify kernel census (GPU1, draft_trace, 30 story rounds w15)
Per round (steady state, from chrome trace key_averages):
- exl3_gemm 16x16x512   x128/round @86.6us = 11.1ms  (target body, 62L)
- exl3_gemm 16x32x256   x 64/round @101.6us = 6.5ms
- exl3_mgemm 16x16x512  x 48/round @88.3us = 4.2ms   (fused qkv groups)
- GDN scan_128          x 48/round @68.3us = 3.3ms   (48 GDN layers)
- exl3_gemm 16x32x128   x 64/round @48.7us = 3.1ms
- conv1d_update         x 48/round @14.2us = 0.7ms; fused_op x48 @5us = 0.24ms
- top-k/gather/scatter tails ~0.3ms total
Top-7 sum ~29ms GPU/round vs measured round 42-50ms => ~13-20ms host/gap overhead per round
(launch storms + sync points), consistent with ~703-launch anatomy. Full timeline JSON kept
(draft_trace.json 115MB) for phase segmentation (draft vs verify windows) next.

## Blocker 1 — replay divergence hunt (2026-11-12 night, GPU2)
- DBLREPLAY (two replays same round): identical logits — consistent with base-row/scratch-row design.
- SPROBE (live conv-ring + state row hashes around replays): verify graph writes live state rows
  per parity (row0 on A, row1 on B); conv-ring changes confounded by async commit drain.
- LAYERDIFF: embed exact; divergence grows through stack (b2 attn ~1-4 -> final hidden ~300-500);
  attention-side contributes more than MLP.
- ATTNPROBE: cache_seqlens/positions arena refresh CORRECT on replay rounds; slots/parity
  consistent via graph keys; rope passes positions tensor.
- CONCLUSION: replay divergence lives in the GDN state-advance handoff (conv ring / scan rows /
  spec staging) between captured verify replays and the eager commit/rewind machinery across
  parity alternation. NEXT (single focused experiment): compare spec_* staging contents +
  ring state after graph replay vs after eager forward for the same round; then either make
  the verify pass stateless-in-graph (state advance only via eager rewind, which already
  recomputes from staging) or fix the handoff directly.
- Trace segmentation (segment_trace.py): verify ~31ms pure-kernel + draft ~11ms pure-kernel;
  ~3200 launches/round total -> launch-storm overhead is a first-class target after graphs.

## Bug 8 — corrupt image bytes killed the connection (ECONNRESET storms)
2026-11-12 night. Root cause verified deterministically by the client-side
investigation: a truncated/corrupt JPEG in `messages` crashed the server during
image decode; the connection died with no error body; the client retried with
the same bad bytes still in history -> repeated deaths (~10x storms). Any valid
image right after a crash worked, proving the server recovered between storms.

### Fixes (serve/serve_prose.py)
1. **`prepare_images()`** decodes and embeds ALL images before any response
   bytes go out (both streaming and non-streaming). A bad image raises
   `ImageDataError(messages_index)`.
2. Handler maps it to `HTTP 400 {"error": "invalid image data at messages[i]"}`
   — verified live: truncated JPEG (1/3 of a real JPEG) -> 400 + server alive;
   valid JPEG immediately after -> 200 with a correct vision answer (95.8 t/s).
3. **Catch-all principle** (the underlying disease): every lane-internal error
   now surfaces as data instead of a dead connection — non-streaming gets
   `500 {"error": "<Type>: <msg>"}`, streaming gets an SSE error event + clean
   chunked terminator. The client can finally SEE why a request failed.
4. `embeddings` computed once and passed into `generate()` (no double decode).

### Launch settings (bw_reps per-class windows applied)
- New `--window N` CLI flag overrides the lane default drafter window.
- New per-request `"draft_window": 8|15` field (validated 1..31) with
  self-healing reset — measured best: prose w8 60.2 t/s, code w15 143.9 t/s.
- Production relaunched: `--window 8` (prose-majority traffic); send
  `"draft_window": 15` for code-heavy agent tasks.

## Blocker 1 — two-world diff night 2 (2026-11-13 early)
Method: run identical generation twice (world E = all-eager GPU1, world G = verify
graphs GPU2), hash layer-0 state (ring/row0/row1/KV/spec_conv/spec_beta) after
every round; no in-check eager (pollution-free, unlike STATECMP).

Findings ladder:
1. WITHOUT the replay-branch bookkeeping, ring/row0 FREEZE from the first pure
   replay round -> FIX: call advance_recurrent_states(batch_ids, params, model)
   after every replay (the forward's host bookkeeping: position, last_history,
   spec_row, spec_shape, post_advance — the commit/rewind machinery depends on
   it and silently no-ops when stale). State moves again after the fix.
2. With the fix: capture rounds (record + replay-once) are BIT-EXACT vs eager
   (state hashes match every capture round, deterministic across runs). First
   STEADY-STATE replay breaks: acceptance collapses to ~1, spec_beta freezes at
   exactly the capture-round value (514.81/524.98 across runs), ring freezes
   soon after; KV pages stay correct; sc (spec_conv) keeps moving (commit still
   consuming). Both A- and B-parity graphs broken in steady state.
3. Interpretation: replays re-execute kernels whose non-tensor inputs (scalar
   kernel args derived from .item()/int() reads — positions, seqlens, offsets)
   were baked at record time; tensor-arg reads (arena, slots, ring, rows) are
   live. Capture rounds are always correct because the baked values ARE that
   round's values — which also explains why earlier "capture-round bit-exact"
   checks passed while steady replays diverged.
NEXT: audit the verify forward path for scalar kernel args (grep .item()/int(
in gated_delta_net chain path, attn/cache get_kv, conv1d update) and convert
them to tensor-driven args (the planned static-state capture surgery).

## Live server bench (2026-11-13, GPU0, commit after Bug-8 fixes)
Launch: `--window 8` (bw_reps prose-best), lane=dspine dynamic c0.25, thinking ON,
3 reps each, 500-1000-token generations, stats from the server itself:
- prose (creative story): 75.2 / 77.5 / 78.3 / 78.4 t/s (1012 tokens)
- code (LRU cache impl): 117.4 / 117.7 / 118.1 t/s (702 tokens)
- structure (strict JSON): 165.3 / 165.3 / 166.0 / 169.6 t/s (157 tokens)
- native tool_calls (booking): 119.1 / 120.7 / 122.6 t/s (129 tokens)
Prose is ~38% over the campaign-close mean (55-57). Removed the per-request
`draft_window` field: the generator sizes buffers from the window at init, so
switching to a larger window at runtime crashes (tensor size mismatch 8 vs 15);
window is launch-time only (`--window`).

## Blocker 1 — SOLVED (2026-11-13): verify-graph replays bit-exact and faster
Full root-cause chain, peeled by two-world diff + zero-tests + graph DOT dumps:
1. Replay rounds skipped the forward's host bookkeeping -> commit machinery
   no-oped (state froze). Fix: advance_recurrent_states on every replay.
2. Hyperconnection cross-pass relays (params["hc_pre"] carry, params["hc_pending"]
   deferred in-place apply) baked stale capture-era tensors. Fix: relay hc_pre
   in/out around each replay, pre-flush pending before capture, run the graph's
   own pending after replay.
3. THE FREEZE: the embedding table is CPU-resident; the verify forward stages
   its (1,16,5120) fp32 output through a persistent PINNED buffer whose H2D
   memcpy is baked into the graph. The buffer is persistent but only the eager
   forward refilled it — every replay re-uploaded the capture round's
   embeddings: frozen x -> frozen b/a/beta (spec_beta constant to the digit)
   while state-dependent paths (conv staging, logits) kept moving. Fix: eager
   refill of the pinned staging right before each replay + never evict a staged
   buffer a live graph references (latent UAF).
Verification: two-world diff 12/12 rounds bit-exact, generation lengths equal.
Engine bench (w15, dynamic c0.25, 3 reps, 420-token gens):
- prose: graph-off 45.2/52.4/54.3 (mean 50.6) -> graph-on 53.2/53.8/58.3 (mean 55.1)
- code: graph-off 92.5/105.5/112.9 (mean 103.6) -> graph-on 109.3/112.0/122.6 (mean 114.6)
Note: capture currently requires EXL3_GDN_FORCE_TORCH=1 (BC fused GDN path uses
its own nested graphs — illegal under capture) + EXL3_BC_ATTN=0; the +9/+11% is
AFTER paying that slower-kernel penalty. Remaining upside: capture-safe fused
GDN/attention path, drafter-side graph (P2), trees (P3).
