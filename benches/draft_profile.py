"""Blocker-2 investigation (GPU1): where do DSpine's 9-10.5 draft ms go?
Instance-level monkeypatch timers around every drafter method + torch.profiler window."""
import os, sys, time, json, atexit
import torch
sys.path.insert(0, os.path.expanduser("~/exl3_pipeline/exllamav3"))
os.environ.setdefault("DSPINE_GRAPH", "1")
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
model.load(draft_model=draft_model, progressbar=False, max_batch_size=1,
           autosplit_no_forward=True, max_chunk_size=2048)
if not draft_cache.initialized:
    draft_model._mark_caches_initialized()
torch.cuda.empty_cache()

# ---- instance-level timers on every drafter method ----
TIMES = {}

def _wrap(obj, name):
    orig = getattr(obj, name).__func__ if hasattr(getattr(obj, name), "__func__") else getattr(obj, name)
    import types
    def timed(self, *a, **k):
        ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        ev0.record()
        try:
            return orig(self, *a, **k)
        finally:
            ev1.record()
            torch.cuda.synchronize()
            d = TIMES.setdefault(name, [0.0, 0.0, 0])
            d[0] += time.perf_counter() - t0
            d[1] += ev0.elapsed_time(ev1) / 1000.0
            d[2] += 1
    setattr(obj, name, types.MethodType(timed, obj))

for m in ("forward", "sample_from_state", "_tail_candidates", "_ensure_tables",
          "_staged_input_block", "_lattice_capture", "_tail_graph_capture",
          "update_kv_from_target", "prefill", "_write_math", "write",
          "_score_math"):
    if hasattr(draft_model, m):
        _wrap(draft_model, m)
# also wrap the ACI injector object if reachable
for attr in ("injector", "aci", "bridge"):
    sub = getattr(draft_model, attr, None)
    if sub is not None and not isinstance(sub, (torch.Tensor, dict)):
        for m in ("write", "_write_math", "next_norm_weight", "layer_read_weight"):
            if hasattr(sub, m):
                _wrap(sub, m)

@atexit.register
def _dump():
    if not TIMES:
        return
    rows = sorted(TIMES.items(), key=lambda kv: -kv[1][0])
    print("\n=== DRAFT METHOD TIMERS (wall_s | cuda_s | calls | ms/call) ===")
    for name, (w, c, n) in rows:
        print(f"{name:28s} wall={w*1000:8.1f}ms cuda={c*1000:8.1f}ms n={n:5d} per_call={w/max(1,n)*1000:6.2f}ms")

story = open(os.path.expanduser("~/exl3_pipeline/conv_prose_story.txt")).read().strip()
gen = Generator(model=model, cache=cache, tokenizer=tokenizer, draft_model=draft_model,
                draft_cache=draft_cache, num_draft_tokens=15, record_draft_stats=True)
prompt = ("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
          f"<|im_start|>user\n{story}<|im_end|>\n<|im_start|>assistant\n")
input_ids = tokenizer.encode(prompt, encode_special_tokens=True)
wjob = Job(input_ids=input_ids, max_new_tokens=24, decode_special_tokens=True, temperature=0.0,
           stop_conditions=[tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")])
gen.enqueue(wjob)
while gen.num_remaining_jobs():
    gen.iterate()

# profiler window: 12 rounds mid-run
from torch.profiler import profile, ProfilerActivity
job = Job(input_ids=input_ids, max_new_tokens=400, decode_special_tokens=True, temperature=0.0,
          stop_conditions=[tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")])
gen.enqueue(job)
rounds = 0
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    while gen.num_remaining_jobs() and rounds < 500:
        gen.iterate()
        rounds += 1
        if rounds == 12:
            break
print(f"[profiled {rounds} rounds]")
prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=45)
print("PROF_DONE")
