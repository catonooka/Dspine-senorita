#!/bin/bash
CUDA_VISIBLE_DEVICES=1 DSPINE_VERIFY_GRAPH=1 DSPINE_VERIFY_GRAPH_DEBUG=1 EXL3_GDN_FORCE_TORCH=1 EXL3_GRAPH_SAFE=1 DSPINE_GRAPH=1 /home/aisever/exl3-env/bin/python - <<'PYEOF' > /home/aisever/exl3_pipeline/vg_probe2.log 2>&1
import os, sys, time, json
import torch
sys.path.insert(0, os.path.expanduser("~/exl3_pipeline/exllamav3"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
drafter_dir = os.path.expanduser("~/exl3_pipeline/output_exl3/Drafter-DSpine-EXL3-5.0bpw-h8")
model_dir = os.path.expanduser("~/exl3_pipeline/output_exl3/Model-27B-EXL3-4.0bpw-h8")
config = Config.from_directory(model_dir)
tokenizer = Tokenizer.from_config(config)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=32768, max_history=16, max_batch_size=1, recurrent_dtype=torch.half)
draft_config = Config.from_directory(drafter_dir)
draft_model = Model.from_config(draft_config, component="text")
draft_cache = Cache(draft_model, max_num_tokens=32768, max_history=16, max_batch_size=1, recurrent_dtype=torch.half)
draft_model.load(progressbar=False, max_batch_size=1, autosplit_no_forward=True, max_chunk_size=2048)
model.load(draft_model=draft_model, progressbar=False, max_batch_size=1, autosplit_no_forward=True, max_chunk_size=2048)
if not draft_cache.initialized:
    draft_model._mark_caches_initialized()
torch.cuda.empty_cache()
prompt_text = open(os.path.expanduser("~/exl3_pipeline/bench_code.txt")).read().strip()
gen = Generator(model=model, cache=cache, tokenizer=tokenizer, draft_model=draft_model,
                draft_cache=draft_cache, num_draft_tokens=15, record_draft_stats=True)
prompt = ("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
          f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n")
input_ids = tokenizer.encode(prompt, encode_special_tokens=True)
job = Job(input_ids=input_ids, max_new_tokens=200, decode_special_tokens=True, temperature=0.0,
          stop_conditions=[tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")])
gen.enqueue(job)
out, t0, rounds_seen = "", None, 0
while gen.num_remaining_jobs():
    res = gen.iterate()
    rounds_seen += 1
    if t0 is None and out:
        t0 = time.time()
    for r in res:
        out += r.get("text", "")
dt = max(1e-6, time.time() - t0)
ntok = len(tokenizer.encode(out, encode_special_tokens=True).flatten())
stats = job.draft_stats or []
acc = sum(a for _, _, a in stats)
n = len(stats)
print(f"RESULT tokens={ntok} rounds={n} committed={round(acc/max(1,n)+1,2)} tps={round(ntok/dt,1)} round_ms={round(dt/max(1,n)*1000,1)} graphs={len(gen._vg_graphs)}")
print("DONE")
PYEOF
