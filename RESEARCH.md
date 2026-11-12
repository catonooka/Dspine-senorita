# Research: solutions to the four measured blockers (2026-11-12)

Sources: GitHub (vLLM, SGLang, TensorRT-LLM, exllamav3, Tencent/DSpine, z-lab/dflash,
llama.cpp, vllm-exl3, JetSpec) + arXiv. Research only — nothing below is implemented yet.

## Blocker 1 — verify pass 26 ms fixed cost; CUDA-graph capture dies (AcceleratorError)

**Verdict: solved problem in production stacks; possibly already fixed upstream in exllamav3.**

- vLLM/SGLang/TRT-LLM all capture the spec-decode verify pass as a PADDED UNIFORM
  decode graph. Requirements: KV page pool + recurrent-state pool PRE-ALLOCATED at
  engine init (no allocation ever inside forward), static block-table/state-slot
  tensors refreshed in place before each replay, row-count buckets (e.g. {2,4,8,16}),
  pad rows writing null state + scratch output, byte-identical warmup, replay guard
  keyed on KV page count. [vLLM cuda_graphs design doc; SGLang runner/backend blog
  2026-08-17 + PR #23906; TRT-LLM piecewise-cudagraph docs + PR #15556]
- vLLM PR #61528 (Mamba2 in piecewise graphs): a hybrid-recurrent mixed step fell
  24.9→10.3 ms, 336→105 eager kernels — same signature as our 703-launch/26 ms verify.
- vLLM PR #61952: GDN core captured branch-free with padded null-state rows — the
  exact blueprint for our 48 GDN layers.
- **exllamav3 itself: issue #501 hit the IDENTICAL `torch.AcceleratorError` under
  graph capture (v1.6.1) — root cause was warmup/capture divergence (warmup scanned
  only [0, t_scan) instead of full width); fixed on dev, with `EXL3_DSA_TOPK_SPLIT=0`
  as workaround.** Issue #230: `ext.hgemm` captures cleanly under `torch.cuda.CUDAGraph`
  single-device. Issue #457 / PR #489 (commit 3a8e2fb): upstream added GDN spec-decode
  accepted-input replay, and "verify replayed through exllamav3's BC graph was the
  fastest mode measured" (92.3 vs 89.2 tok/s) — i.e. **upstream already has a working
  graph path for exactly this**. Our clone is v1.6.6; these may be newer on dev —
  first action when we code: `git fetch` and diff.
- torch.compile cannot trace exl3_gemm (issue #64) — dynamic-graph patchable-args is
  the framework's own mechanism; raw CUDAGraph capture works (issue #230).
- Endgame papers: MPK/Mirage persistent kernels (2512.22219), FlashFormer (2505.22758),
  "One Megakernel is Not Enough" (2610.35336) — per-region megakernels; launch overhead
  alone is 14.6% of e2e decode (Ada-MK 2605.11581).

Expected if ported: verify 26.3→4-7 ms (2 rows) / 7-9 ms (16 rows) → prose ~90-95 t/s,
code ~150 t/s at current acceptance. Effort: medium (1-2 weeks), or near-zero if the
upstream dev branch already carries it.

## Blocker 2 — DSpine draft pass 2× DFlash2 (8.7-10.5 ms vs 4.3-4.4 ms)

**Verdict: our port's kernelization, not the architecture. Paper's DSpine-specific cost is 0.15 ms/round.**

- DSpine paper (arXiv 2609.36173 Fig. 6, Qwen3-8B/SGLang): parallel injection adds
  ≤0.15 ms per round across all 5 layers; transition-cached decoding costs less than
  half of DSpark-style correction heads.
- Bandwidth floor on a 3090 (936 GB/s): 5-layer backbone 1.65B params = 1.10 ms at
  ~5bpw; lm_head [16,5120]×[5120,248320] = 2.72 ms BF16 / 1.36 ms FP8 / 0.85 ms at 5bpw;
  codebook 0.51 GB BF16 = 0.54 ms if fully streamed. Our 8.7-10.5 ms sits 4-5× above
  the ~2 ms floor → the gap is launch overhead + unfused ops + likely re-streaming the
  2.54 GB BF16 head every round.
- Official SGLang plugin (github.com/Tencent/DSpine): transition tables TG/TSV built
  ONCE at startup (build_serial_tables); per round only gathers ~0.74 MB TG/TSV rows +
  ~2.5 MB lm_head rescore; draft head = FP8-coarse full-vocab retrieval (flashinfer
  top-k) + BF16 rescore of top-16 rows only; the whole draft runs inside a captured
  CUDA graph (DSpineDraftCudaGraphRunner). Plugin refuses quantized draft weights
  (BF16-only official path).
- Third-party confirmation: llama.cpp PR #31941 — "The 27B draft step costs more and
  offsets its higher tau" (DSpine vs DFlash2 at 27B).
- **Cheapest lever found: `DSpineDraftModel.set_block_size` officially supports 2..16
  on a b16-trained checkpoint.** z-lab/dflash issue #151: on memory-bound hardware,
  block 5-6 beat block 16 by 24-29% e2e despite lower acceptance. Block 8 halves tail
  ops + verify rows; DFlash2 ships block 8 for exactly this reason.
- Head GEMM kernel: Marlin (arXiv 2408.11743) is Ampere-native w4a16, designed for
  batch ~16-32 — exactly the [16,5120]×[5120,248320] shape. Machete is Hopper-only.

Ranked: (1) port/copy official tail (once-built tables, FP8/4bpw coarse head, graph)
→ ~3-5 ms; (2) block_size 8 → instant ~half tail+state cost, sweep 8/12/16;
(3) Marlin head → 0.6-0.9 ms head streaming.

## Blocker 3 — dynamic verify-width truncation never fires (139/139 full width)

**Verdict: not a bug and not a lever at batch 1. Stop pursuing.**

- DSpark scheduler (arXiv 2607.05147): truncates only when Θ = tau × SPS(B) drops,
  where SPS(B) is the engine's steps-per-second-vs-batch curve. On a memory-bound GPU
  at batch 1, SPS(B) is FLAT for B=1..16 (same weight stream per round) → every row
  with positive survival probability raises Θ → full-width verify is the OPTIMAL
  schedule. Our histogram is correct behavior.
- DSpine's own policy: uniform verify width for batch 1-7; per-row confidence routing
  exists only in the batch-8 ragged budget; WITHOUT a tune_sps profile it full-blocks
  by design (docs/sps_tuning.md). Their shipped calibration histogram is from
  agent/code data — wrong domain for story text anyway.
- Any truncation we force just reimplements a smaller static block — which
  `set_block_size` does natively (Blocker 2, lever 2).

## Blocker 4 — acceptance ~2.9 committed/round on creative prose

**Verdict: ~3 is the published FIELD CEILING for chain drafting; >5 exists only with trees + single-pass verify.**

- Chain drafters on Qwen3-class targets, MT-Bench, T=0: EAGLE-3 tau = 2.83 (DFlash
  paper table) / 3.19-3.62 (JetSpec eval); DFlash ~3.2-3.4; Hydra 3.65 (Vicuna);
  our DSpine 2.88 / DFlash2 2.83-3.04 on harder long-form story = in-family.
  No published chain drafter beats ~3.6 on open-ended text at this scale.
- >5 committed/round IS published, both without retraining the backbone:
  - **DDTree (arXiv 2604.12989): builds a best-first TREE from DFlash's per-position
    distributions** (which DFlash2 already computes) — DFlash 5.38×→7.35× on AIME24,
    tau 10.42; MT-Bench tau 5.35 at 4.26× on Qwen3-30B-A3B. No new drafter training.
  - JetSpec (arXiv 2606.18394, github hao-ai-lab/JetSpec): trained 5-layer causal
    tree head, MT-Bench tau 5.59-5.9, up to 9.64× on MATH-500; head ~1-2 GB; needs
    training on ~800K target-generated samples; no public 27B dense checkpoint.
  - JetSpec++ (2610.19792): tau 8.94 avg incl. MT-Bench with pipelined draft/verify —
    the headroom ceiling, GB200-class serving.
- Tree budget economics: acceptance grows monotonically with node budget but e2e
  speedup PEAKS at 256-512 nodes (DDTree; SMART 2604.09731 formalizes the marginal
  benefit-cost rule; OPT-Tree 2406.17276).
- Tree-verify kernels for launch-bound rounds: FlashInfer branch/tree-attention
  (Apache-2.0, portable), P-EAGLE fused Triton batch-prep (AWS/vLLM, explicitly
  built to kill per-step launch overhead), DeFT-Flatten fused tree attention
  (2404.00242, 73-99% KV-IO cut). Our 240 ms/2500-launch tree rounds are an unfused
  implementation of a solved problem; estimate 60-90 ms with single-pass verify.
- n-gram/retrieval drafting on story: PLD tau 1.63 MT-Bench / 1.19 Alpaca vs 2.50
  CNN-DM — pays only on repetition spans; our hybrid8 already captures most of it.
  SAM-Decoding +3-11% on EAGLE-2 as a cheap overlay.
- Do not chase: "Exponential Speculative Sampling", "JANUS", "DIAMOND", "GLIDER",
  "Hydra-2" as separate artifacts — no such papers/repos exist (checked arXiv +
  OpenReview; nearest real: Exponential Races 2504.15475 theory-only, GliDe 2502.17421,
  GRIFFIN 2502.11018).

## Combined path to the targets (when we start coding again)

1. `git fetch` exllamav3 dev; test in-tree BC-graph path + EXL3_DSA_TOPK_SPLIT=0 with
   our verify capture (Blocker 1 may collapse to config).
2. `set_block_size 8` + width sweep (instant, halves DSpine tail cost).
3. Port official DSpine tail (once-built TG/TSV, coarse head, draft graph) → 3-5 ms.
4. With graph verify (~7-9 ms) + fast draft (~4 ms): round ≈ 20-25 ms → prose ~75-95,
   code ~150-170 at current acceptance.
5. Trees (DDTree selection from existing marginals + FlashInfer-style fused verify)
   are the only published route past tau≈3 on creative text → the real 120-prose play.
