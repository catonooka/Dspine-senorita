"""Floor + verify-width scaling: dense decode cost, and round cost vs draft width."""
import os, sys, time, json
import torch
sys.path.insert(0, os.path.expanduser("~/exl3_pipeline/exllamav3"))
os.environ.setdefault("DSPINE_GRAPH", "1")
os.environ["DSPINE_ROUND_TIMER"] = "1"
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

user = open(os.path.expanduser("~/exl3_pipeline/conv_prose_story.txt")).read().strip()
prompt = ("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
          f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")

def run(tag, use_draft, ndt):
    gen = Generator(model = model, cache = cache, tokenizer = tokenizer,
                    draft_model = draft_model if use_draft else None,
                    draft_cache = draft_cache if use_draft else None,
                    num_draft_tokens = ndt, record_draft_stats = use_draft)
    # warmup lane (graphs/compiles), untimed
    wjob = Job(input_ids = tokenizer.encode(prompt, encode_special_tokens = True),
               max_new_tokens = 24, decode_special_tokens = True, temperature = 0.0,
               stop_conditions = [tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")])
    gen.enqueue(wjob)
    while gen.num_remaining_jobs():
        gen.iterate()
    gen._rt_on = None; gen._rt_draft = []; gen._rt_gen = []
    input_ids = tokenizer.encode(prompt, encode_special_tokens = True)
    job = Job(input_ids = input_ids, max_new_tokens = 220, decode_special_tokens = True,
              temperature = 0.0,
              stop_conditions = [tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")])
    gen.enqueue(job)
    gen._rt_on = None; gen._rt_draft = []; gen._rt_gen = []
    t0 = None; out_text = ""
    while gen.num_remaining_jobs():
        results = gen.iterate()
        if t0 is None and (not use_draft or gen._rt_draft):
            t0 = time.time()
        for r in results:
            out_text += r.get("text", "")
    ntok = len(tokenizer.encode(out_text, encode_special_tokens = True).flatten())
    dt = time.time() - t0 if t0 else -1
    if use_draft:
        d = gen._rt_draft; g = gen._rt_gen; n = min(len(d), len(g))
        md = sum(d[:n]) / n * 1000; mg = sum(g[:n]) / n * 1000
        stats = job.draft_stats or []
        committed = sum(a for _, _, a in stats) / max(1, len(stats))
        out = dict(rounds = n, committed_per_round = round(committed, 2),
                   round_ms = round(dt / max(1, n) * 1000, 1),
                   draft_ms = round(md, 1), gen_ms = round(mg, 1), tps = round(ntok / dt, 1))
    else:
        out = dict(round_ms = round(dt / max(1, ntok) * 1000, 1), tps = round(ntok / dt, 1))
    print(f"[{tag}] {json.dumps(out)}", flush = True)
    return out

report = {}
report["dense_nodefault"] = run("dense", False, 0)
for ndt in (1, 4, 8, 15):
    report[f"ndt{ndt}"] = run(f"ndt{ndt}", True, ndt)
json.dump(report, open(os.path.expanduser("~/exl3_pipeline/floor_width.json"), "w"), indent = 2)
print("DONE")
