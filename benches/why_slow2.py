"""Why-is-DSpine-slow #2 (GPU2): target verify cost vs width curve + DFlash2 reference."""
import os, sys, time, json
import torch
sys.path.insert(0, os.path.expanduser("~/exl3_pipeline/exllamav3"))
os.environ.setdefault("DSPINE_GRAPH", "1")
os.environ["DSPINE_ROUND_TIMER"] = "1"
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job

model_dir = os.path.expanduser("~/exl3_pipeline/output_exl3/Model-27B-EXL3-4.0bpw-h8")
story = open(os.path.expanduser("~/exl3_pipeline/conv_prose_story.txt")).read().strip()

def load_pair(drafter_dir):
    config = Config.from_directory(model_dir)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=32768, max_history=16, max_batch_size=1, recurrent_dtype=torch.half)
    dc = Config.from_directory(drafter_dir)
    dm = Model.from_config(dc, component="text")
    dcache = Cache(dm, max_num_tokens=32768, max_history=16, max_batch_size=1, recurrent_dtype=torch.half)
    dm.load(progressbar=False, max_batch_size=1, autosplit_no_forward=True, max_chunk_size=2048)
    model.load(draft_model=dm, progressbar=False, max_batch_size=1,
               autosplit_no_forward=True, max_chunk_size=2048)
    if not dcache.initialized:
        dm._mark_caches_initialized()
    torch.cuda.empty_cache()
    return model, cache, tokenizer, dm, dcache

def anatomy(tag, model, cache, tokenizer, dm, dcache, ndt, dynamic, conf, max_new=300):
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                    draft_model=dm, draft_cache=dcache,
                    num_draft_tokens=ndt, record_draft_stats=True,
                    dynamic_draft_tokens=dynamic, draft_confidence=conf)
    prompt = ("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
              f"<|im_start|>user\n{story}<|im_end|>\n<|im_start|>assistant\n")
    input_ids = tokenizer.encode(prompt, encode_special_tokens=True)
    wjob = Job(input_ids=input_ids, max_new_tokens=24, decode_special_tokens=True, temperature=0.0,
               stop_conditions=[tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")])
    gen.enqueue(wjob)
    while gen.num_remaining_jobs():
        gen.iterate()
    gen._rt_draft, gen._rt_gen = [], []
    job = Job(input_ids=input_ids, max_new_tokens=max_new, decode_special_tokens=True, temperature=0.0,
              stop_conditions=[tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")])
    gen.enqueue(job)
    while gen.num_remaining_jobs():
        gen.iterate()
    d, g = gen._rt_draft, gen._rt_gen
    n = min(len(d), len(g))
    stats = job.draft_stats or []
    accs = [s[2] + 1 for s in stats]
    rep = dict(
        rounds=n, committed=round(sum(accs) / max(1, len(accs)), 2),
        draft_ms=round(sum(d[:n]) / n * 1000, 1),
        gen_ms=round(sum(g[:n]) / n * 1000, 1),
        round_ms=round((sum(d[:n]) + sum(g[:n])) / n * 1000, 1),
    )
    rep["proj_tps"] = round(rep["committed"] / rep["round_ms"] * 1000, 1)
    print(f"[{tag}] {json.dumps(rep)}", flush=True)
    return rep

report = {}
# Part A: width-cost curve with DSpine drafter (window native 16), static widths
model, cache, tokenizer, dm, dcache = load_pair(
    os.path.expanduser("~/exl3_pipeline/output_exl3/Drafter-DSpine-EXL3-5.0bpw-h8"))
for ndt in (0, 1, 3, 7, 15):
    tag = f"ds_w{ndt}"
    rep = anatomy(tag, model, cache, tokenizer, dm, dcache, ndt, False, 0.25)
    report[tag] = rep
del model, cache, dm, dcache
torch.cuda.empty_cache()
# Part B: DFlash2 EXL3 reference at native window 7 (diagnostic only)
model, cache, tokenizer, dm, dcache = load_pair(
    os.path.expanduser("~/exl3_pipeline/output_exl3/Drafter-DFlash2-EXL3-5.0bpw-h8"))
report["df2_w7_static"] = anatomy("df2_w7_static", model, cache, tokenizer, dm, dcache, 7, False, 0.25)
report["df2_w7_dyn_c025"] = anatomy("df2_w7_dyn_c025", model, cache, tokenizer, dm, dcache, 7, True, 0.25)
json.dump(report, open(os.path.expanduser("~/exl3_pipeline/why_slow2.json"), "w"), indent=2)
print("DONE")
