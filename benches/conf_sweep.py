"""Dynamic-verify confidence sweep: full-block draft + calibrator-truncated verify.
Env: DRAFTER_DIR (raw_drafter or raw_dflash2), WINDOW (15 or 7), OUT_JSON.
Story prompt, think + no-think (closed-think trick), greedy, 400 tok each."""
import os, sys, time, json
import torch
sys.path.insert(0, os.path.expanduser("~/exl3_pipeline/exllamav3"))
os.environ.setdefault("DSPINE_GRAPH", "1")
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job

drafter_dir = os.path.expanduser(os.environ["DRAFTER_DIR"])
window = int(os.environ.get("WINDOW", "15"))
out_json = os.path.expanduser(os.environ["OUT_JSON"])

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

user = open(os.path.expanduser("~/exl3_pipeline/conv_prose_story.txt")).read().strip()

def run(tag, conf, think):
    prompt = ("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
              f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
    if not think:
        prompt = prompt + "<think>\n\n</think>\n\n"
    input_ids = tokenizer.encode(prompt, encode_special_tokens=True)
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                    draft_model=draft_model, draft_cache=draft_cache,
                    num_draft_tokens=window, record_draft_stats=True,
                    dynamic_draft_tokens=True, draft_confidence=conf)
    wjob = Job(input_ids=input_ids, max_new_tokens=24, decode_special_tokens=True,
               temperature=0.0,
               stop_conditions=[tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")])
    gen.enqueue(wjob)
    while gen.num_remaining_jobs():
        gen.iterate()
    job = Job(input_ids=input_ids, max_new_tokens=400, decode_special_tokens=True,
              temperature=0.0,
              stop_conditions=[tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")])
    gen.enqueue(job)
    out = ""
    t0 = None
    while gen.num_remaining_jobs():
        results = gen.iterate()
        if t0 is None and out:
            t0 = time.time()
        for r in results:
            out += r.get("text", "")
    dt = max(1e-6, time.time() - t0)
    ntok = len(tokenizer.encode(out, encode_special_tokens=True).flatten())
    stats = job.draft_stats or []
    n = len(stats)
    acc_tok = sum(a for _, _, a in stats)
    rep = dict(tokens=ntok, rounds=n,
               committed=round(acc_tok / max(1, n) + 1, 2),
               tps=round(ntok / dt, 1))
    print(f"[{tag}] {json.dumps(rep)}", flush=True)
    return rep

report = {}
confs = tuple(float(x) for x in os.environ.get("CONFS", "0.25,0.4,0.5,0.65").split(","))
for think in (True, False):
    mode = "think" if think else "nothink"
    for conf in confs:
        tag = f"{mode}_dyn{window}_c{conf}"
        report[tag] = run(tag, conf, think)
json.dump(report, open(out_json, "w"), indent=2)
print("DONE")
