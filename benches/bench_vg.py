"""Noise-band quantification: 3 reps × best configs, 400 tok, same process."""
import os, sys, time, json
import torch
sys.path.insert(0, os.path.expanduser("~/exl3_pipeline/exllamav3"))
os.environ.setdefault("DSPINE_GRAPH", "1")
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job

drafter_dir = os.path.expanduser("~/exl3_pipeline/output_exl3/Drafter-DSpine-EXL3-6.0bpw-h8")
model_dir = os.path.expanduser("~/exl3_pipeline/output_exl3/Model-27B-EXL3-4.0bpw-h8")
config = Config.from_directory(model_dir)
tokenizer = Tokenizer.from_config(config)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=32768, max_history=16, max_batch_size=1, recurrent_dtype=torch.half)
draft_config = Config.from_directory(drafter_dir)
draft_model = Model.from_config(draft_config, component="text")
draft_cache = Cache(draft_model, max_num_tokens=32768, max_history=16, max_batch_size=1, recurrent_dtype=torch.half)
draft_model.load(progressbar=False, max_batch_size=1, autosplit_no_forward=True, max_chunk_size=2048)
model.load(draft_model=draft_model, progressbar=False, max_batch_size=1,
           autosplit_no_forward=True, max_chunk_size=2048)
if not draft_cache.initialized:
    draft_model._mark_caches_initialized()
torch.cuda.empty_cache()


PROMPTS = {
    "code": open(os.path.expanduser("~/exl3_pipeline/bench_code.txt")).read().strip(),
    "story": open(os.path.expanduser("~/exl3_pipeline/conv_prose_story.txt")).read().strip(),
}
import time, sys
ON = sys.argv[1] == "1" if len(sys.argv) > 1 else True
gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                draft_model=draft_model, draft_cache=draft_cache,
                num_draft_tokens=15, dynamic_draft_tokens=True, draft_confidence=0.25)
for kind in ("story", "code"):
    tps_l, toks_l = [], []
    for rep in range(3):
        prompt = ("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
                  f"<|im_start|>user\n{PROMPTS[kind]}<|im_end|>\n<|im_start|>assistant\n")
        input_ids = tokenizer.encode(prompt, encode_special_tokens=True)
        job = Job(input_ids=input_ids, max_new_tokens=420, decode_special_tokens=True,
                  temperature=0.0,
                  stop_conditions=[tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")])
        gen.enqueue(job)
        out, t0, n = "", None, 0
        while gen.num_remaining_jobs():
            rs = gen.iterate()
            for r in rs:
                d = r.get("text", "")
                if d:
                    if t0 is None: t0 = time.time()
                    out += d
        dt = time.time() - t0
        n = len(tokenizer.encode(out, encode_special_tokens=True).flatten())
        tps_l.append(round(n / dt, 1)); toks_l.append(n)
    print(f"{'GRAPH-ON ' if ON else 'GRAPH-OFF'} {kind:>5}: tps={tps_l} tokens={toks_l}", flush=True)
