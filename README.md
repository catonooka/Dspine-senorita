# Dspine-senorita

By **catonooka**: [X / Twitter](https://x.com/catonooka) -
[GitHub](https://github.com/catonooka) -
[Hugging Face](https://huggingface.co/catonooka) (target quant:
[Qwen3.8-27B-EXL3-4.0bpw-h8](https://huggingface.co/catonooka/Qwen3.8-27B-EXL3-4.0bpw-h8)).

Creative-prose and code decode speed for **Qwen3.8-27B EXL3 4.0bpw-h8 on a single
RTX 3090**, drafted with **DSpine** (Tencent's parallel 5-layer drafter) - a full
measured campaign: dynamic verification, n-gram hybrid drafting, drafter
re-quantization (EXL3 5.0/6.0bpw, fp16 heads), sparse draft heads, and a
verify-pass CUDA-graph attempt, on top of [exllamav3](https://github.com/turboderp/exllamav3) v1.6.6.

Every number in this repo is measured on the same rig, greedy, single stream,
GPU locked at 350 W / 1770 MHz. Run-to-run spread is stated wherever it matters.

## Results (400-token generations, DSpine drafter only)

| Lane | Config | Mean t/s (×3 reps) | Best single |
|---|---|---:|---:|
| Code | static15 or dyn c0.25 | 131-138 | 143.4 |
| Creative prose (thinking on) | static15 + n-gram hybrid 8 | 55-57 | 64.7 |
| Dense floor (no drafting) | - | 42.4 | - |

**Live server, 2026-11-13 (latest: verify graphs + K8V8 cache adopted).**
Thinking on, `--window 8`, 8-bit K/V cache with a 163,840-token context pool,
verify-pass CUDA graphs. All English prompts, greedy, 2-3 reps, warm server.

| Workload | Peak t/s (bias-tuned prompt) | Typical t/s (hard prompt) |
|---|---:|---:|
| Prose | **96.4** | 60+ |
| Code | **147.3** | 110 |
| Structure (strict JSON) | **234.6** | 110 |
| Math (step-by-step) | **224.7** | 125 |

Continuous multi-turn chat gains 5-13% over the same build without graphs, code
+11%. K8V8 replaced the fp16 cache at equal VRAM: context grew 90k -> 164k tokens
and every class got faster (+11-14%) because each verify pass reads half the KV
bytes. Correctness is covered by the upstream lossless suite (5/5 with graphs on
and off, including the 8-bit-cache decode consistency case).

Reference, same rig and harness (not the project default): DFlash2-EXL3-5.0bpw
drafter reaches ~75 t/s on prose. DSpine keeps the code lane and is the project's
default lane; both drafters ship as switchable lanes in the server.

**Noise band:** ±15-20% run-to-run at identical config. Kernel nondeterminism flips
near-tie argmaxes, which diverges the greedy trajectory, which changes speculative
acceptance (committed/round swings 5.2-6.7 on code, 2.3-3.0 on prose). Any single-run
number under ~150 t/s code / ~65 t/s prose is inside the band.

**Honest ceiling (campaign close-out):** committed-tokens-per-round is the wall -
6.2 (code) / 2.7 (prose) per 42-46 ms round. 200 t/s code needs 8.5 committed or
sub-30 ms rounds; 120 t/s prose needs 5+ committed, which no drafter we tested
reaches on high-entropy creative text. Details and the remaining levers are in
[REPORT.md](REPORT.md).

## Measured and rejected

Half the value of a campaign is what didn't work:

| Lever | Verdict | Why |
|---|---|---|
| Sparse draft head (32k vocab subset) | ✗ rejected | −10% on code (subset drops good draft tokens), neutral on prose |
| fp16 draft head (5.0bpw-h16) | ✗ rejected | +0.9% code ≈ noise, worse prose acceptance; head precision isn't the bottleneck |
| 6.0bpw drafter body | ✗ rejected | +0-3% code, inside noise band; not worth +0.2 GB |
| Verify-pass CUDA graph (first attempt) | fixed later, now default | capture died with `AcceleratorError`; full root-cause chain in REPORT.md; re-landed and adopted in production |
| Hybrid n-gram + dynamic verify stacked | ✗ rejected | n-gram drafts disagree with confidence truncation → wasted verify width |
| Dynamic verify on code | = wash | matches static15 once repeated (earlier single-rep 163.8 was trajectory luck) |

Accepted levers: dynamic verification (DSpine c0.25 = best confidence, U-shaped sweep),
n-gram hybrid as **static + hyb8 on prose** (+5-8%), EXL3 5.0bpw-h8 drafter quant
(equal acceptance to BF16 at 40% the size), **K8V8 KV cache** (+11-14% and 1.8x context at the same VRAM), **verify-pass CUDA graphs** (chat +5-13%, code +11%).

## Repository layout

```
engine/    exllamav3 v1.6.6 campaign diff (one patch) + dspine.py / tree_engine.py
benches/   every harness that produced a reported number, + raw result JSONs
serve/     serve_prose.py - OpenAI-compatible server (stdlib only, no deps)
REPORT.md  the deep report: paper vs rig, camp-by-camp ladder, laws, verdicts
RESEARCH.md  sourced solutions to the four measured blockers (graphs, draft tail, SPS, trees)
```

## Quick start

```bash
# deps: exllamav3 (patched with engine/0001-*.patch), torch + CUDA, a GPU with 24 GB
# production launch used on the bench rig (graphs + K8V8):
#   systemd-run --user --unit=senorita --collect -p MemoryMax=infinity \
#     -E CUDA_VISIBLE_DEVICES=0 \
#     -E DSPINE_GRAPH=1 -E DSPINE_VERIFY_GRAPH=1 \
#     -E EXL3_GRAPHS=0 -E EXL3_GRAPH_SAFE=1 -E EXL3_BC_ATTN=0 \
#     -E DSPINE_DRAFT_GRAPH=1 -E DSPINE_DRAFT_SAMPLE_GRAPH=1 \
#     bash -c 'python serve/serve_prose.py --port 8900 --window 8 \
#       --kv8 --pool 163840 > server.log 2>&1'
# drop the five -E env vars and the --kv8 --pool flags for the no-graphs fp16 build
python serve/serve_prose.py --port 8900 --window 8 --kv8 --pool 163840

curl -s localhost:8900/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"Write a story about a lighthouse."}],
  "max_tokens": 400
}'
```

Server features (all measured, live):

- Lanes: `dspine` (default, window 15, dynamic verify c0.25), `dspine-bf16`, `dflash2`, `dflash2-bf16` - per-lane context pools sized for 24 GB
- `--kv8` quantizes the target KV cache to 8-bit (faster and 1.8x the context at the same
  VRAM); `--pool N` overrides the context pool; `/v1/models` reports the live pool size
- Vision: `image_url` parts (base64 or http); the vision tower loads on first image and auto-unloads
- Thinking: `reasoning_effort` (xhigh default / medium / low), `enable_thinking`, z.ai-style `thinking: {"type": "disabled"}` - invalid values return the chat template's own 400
- Streaming SSE, per-request draft stats, clean 400/503 on oversize prompts and OOM
- `--window N` sets the drafter window at launch (bw_reps: prose w8, code w15; runtime
  per-request switching is NOT supported - the generator sizes buffers at init)
- Corrupt/truncated images return `400 {"error": "invalid image data at messages[i]"}`
  instead of killing the connection; every lane-internal error surfaces as JSON
  (Bug 8 - this was the root cause of the ECONNRESET retry storms)

Reproduce the benches: each script under `benches/` is standalone (`python benches/spinecamp1.py`),
reads its prompts from the pipeline directory, and writes its result JSON next to
the copies in `benches/results/`.

## Credits

- **[turboderp](https://github.com/turboderp) and the exllamav3 contributors** - the
  engine everything here runs on, including the in-tree dynamic-verification and
  n-gram machinery this campaign tuned. The speed baseline (dense 42.4 t/s, the whole
  verify path) is their work.
- **Tencent's DSpine team** - the drafter architecture this project is built around:
  [paper (arXiv 2609.36173)](https://arxiv.org/abs/2609.36173), [GitHub](https://github.com/Tencent/DSpine),
  [HF drafter](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16). Our contribution
  is the deployment-side campaign: quantizing the drafter to EXL3, the verification-width
  and hybrid sweeps, and the negative results.
- **z-lab** - DFlash2, used as the comparison lane throughout.
- The target quant is published at
  [catonooka/Qwen3.8-27B-EXL3-4.0bpw-h8](https://huggingface.co/catonooka/Qwen3.8-27B-EXL3-4.0bpw-h8).

## License

MIT - see [LICENSE](LICENSE). The engine patch in `engine/` modifies exllamav3
(also MIT, © Turboderp); drafter architectures referenced are the property of
their authors. No model weights are included in this repository.
