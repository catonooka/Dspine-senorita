"""Verify-pass profiler (GPU0): where do the ~34ms of a 16-row verify round go."""
import os, sys, time, json
import torch
sys.path.insert(0, os.path.expanduser("~/exl3_pipeline/exllamav3"))
os.environ.setdefault("DSPINE_GRAPH", "1")
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job

model_dir = os.path.expanduser("~/exl3_pipeline/output_exl3/Model-27B-EXL3-4.0bpw-h8")
drafter_dir = os.path.expanduser("~/exl3_pipeline/raw_drafter")
config = Config.from_directory(model_dir)
tokenizer = Tokenizer.from_config(config)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 32768, max_history = 16, max_batch_size = 1, recurrent_dtype = torch.half)
draft_config = Config.from_directory(drafter_dir)
draft_model = Model.from_config(draft_config, component = "text")
draft_cache = Cache(draft_model, max_num_tokens = 32768, max_history = 16, max_batch_size = 1, recurrent_dtype = torch.half)
draft_model.load(progressbar = False, max_batch_size = 1, autosplit_no_forward = True, max_chunk_size = 2048)
model.load(draft_model = draft_model, progressbar = False, max_batch_size = 1,
           autosplit_no_forward = True, max_chunk_size = 2048)
if not draft_cache.initialized:
    draft_model._mark_caches_initialized()
torch.cuda.empty_cache()

generator = Generator(model = model, cache = cache, tokenizer = tokenizer,
                      draft_model = draft_model, draft_cache = draft_cache,
                      num_draft_tokens = 15, record_draft_stats = True)

user = open(os.path.expanduser("~/exl3_pipeline/conv_prose_story.txt")).read().strip()
prompt = ("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
          f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
input_ids = tokenizer.encode(prompt, encode_special_tokens = True)
job = Job(input_ids = input_ids, max_new_tokens = 260, decode_special_tokens = True,
          temperature = 0.0,
          stop_conditions = [tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")])
generator.enqueue(job)
rounds = 0
# warmup 30 rounds unprofiled, then profile 40
from torch.profiler import profile, ProfilerActivity
profiling = False
while generator.num_remaining_jobs():
    if not profiling and rounds == 30:
        profiling = True
        ctx = profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA])
        ctx.__enter__()
    generator.iterate()
    rounds += 1
    if profiling and rounds == 70:
        break
if profiling:
    ctx.__exit__(None, None, None)
    evs = ctx.key_averages()
    rows = sorted(evs, key = lambda e: e.self_device_time_total, reverse = True)
    print("=== TOP GPU TIME (self, ms total over 40 rounds) ===")
    for e in rows[:18]:
        if e.self_device_time_total > 0:
            print(f"{e.self_device_time_total/1000:9.2f} ms  {e.count:5d}x  {e.key[:90]}")
    print("=== TOP CPU (self, ms total) ===")
    for e in sorted(evs, key = lambda e: e.self_cpu_time_total, reverse = True)[:12]:
        if e.self_cpu_time_total > 100000:
            print(f"{e.self_cpu_time_total/1000:9.1f} ms  {e.count:5d}x  {e.key[:90]}")
print("DONE")
