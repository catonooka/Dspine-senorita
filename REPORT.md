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
| dyn c0.25 + sparse32k | 147.1 | 61.9 (dyn) |
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
