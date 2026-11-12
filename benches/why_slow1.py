"""Why-is-DSpine-slow #1 (GPU1): phase anatomy + verify width histogram, serving config."""
import os, sys, time, json, collections
import torch
sys.path.insert(0, os.path.expanduser("~/exl3_pipeline/exllamav3"))
os.environ.setdefault("DSPINE_GRAPH", "1")
os.environ["DSPINE_ROUND_TIMER"] = "1"
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

PROMPTS = {
    "code": open(os.path.expanduser("~/exl3_pipeline/bench_code.txt")).read().strip(),
    "story": open(os.path.expanduser("~/exl3_pipeline/conv_prose_story.txt")).read().strip(),
}

def run(tag, kind, dynamic, conf):
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                    draft_model=draft_model, draft_cache=draft_cache,
                    num_draft_tokens=15, record_draft_stats=True,
                    dynamic_draft_tokens=dynamic, draft_confidence=conf)
    prompt = ("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
              f"<|im_start|>user\n{PROMPTS[kind]}<|im_end|>\n<|im_start|>assistant\n")
    input_ids = tokenizer.encode(prompt, encode_special_tokens=True)
    wjob = Job(input_ids=input_ids, max_new_tokens=24, decode_special_tokens=True,
               temperature=0.0,
               stop_conditions=[tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")])
    gen.enqueue(wjob)
    while gen.num_remaining_jobs():
        gen.iterate()
    gen._rt_draft, gen._rt_gen = [], []
    job = Job(input_ids=input_ids, max_new_tokens=400, decode_special_tokens=True,
              temperature=0.0,
              stop_conditions=[tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")])
    gen.enqueue(job)
    out, t0 = "", None
    while gen.num_remaining_jobs():
        results = gen.iterate()
        if t0 is None and gen._rt_draft:
            t0 = time.time()
        for r in results:
            out += r.get("text", "")
    dt = max(1e-6, time.time() - t0)
    d, g = gen._rt_draft, gen._rt_gen
    n = min(len(d), len(g))
    stats = job.draft_stats or []
    widths = [s[1] for s in stats]
    accs = [s[2] + 1 for s in stats]
    hist = collections.Counter()
    for w in widths:
        hist["1-4" if w <= 4 else "5-8" if w <= 8 else "9-12" if w <= 12 else "13-16"] += 1
    rep = dict(
        tps=round(len(out and tokenizer.encode(out, encode_special_tokens=True).flatten()) / dt, 1),
        rounds=n, committed=round(sum(accs) / max(1, len(accs)), 2),
        round_ms=round(dt / max(1, n) * 1000, 1),
        draft_ms=round(sum(d[:n]) / n * 1000, 1),
        gen_ms=round(sum(g[:n]) / n * 1000, 1),
        verify_width_mean=round(sum(widths) / max(1, len(widths)), 1),
        width_hist=dict(sorted(hist.items())),
    )
    print(f"[{tag}] {json.dumps(rep)}", flush=True)
    return rep

report = {}
report["story_dyn_c025"] = run("story_dyn_c025", "story", True, 0.25)
report["story_static15"] = run("story_static15", "story", False, 0.25)
report["code_static15"] = run("code_static15", "code", False, 0.25)
json.dump(report, open(os.path.expanduser("~/exl3_pipeline/why_slow1.json"), "w"), indent=2)
print("DONE")
