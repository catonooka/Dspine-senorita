# DSpine-senorita

Max creative-prose decode speed for Qwen3.8-27B EXL3 4.0bpw-h8 on one RTX 3090:
DSpine vs DFlash2 drafting, dynamic verification, EXL3 drafter re-quantization.

- **REPORT.md** — the deep report: paper vs rig, measured ladder, laws, verdicts.
- **serve/serve_prose.py** — OpenAI-compatible server, champion config default
  (DFlash2-EXL3-5.0bpw drafter, window 7, dynamic verify, confidence 0.25).
- **engine/** — our exllamav3 v1.6.6 campaign diff (DSpine architecture attach,
  tree verification engine, GDN tree paths) as one patch + the two new files.
- **benches/** — every harness that produced a reported number + raw result JSONs.

Quick start:

```bash
CUDA_VISIBLE_DEVICES=0 ~/exl3-env/bin/python serve/serve_prose.py --port 8900
curl -s localhost:8900/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a story about a lighthouse."}],"max_tokens":400}'
```

Numbers in REPORT.md are measured, same-day A/Bs; projections are labeled.
