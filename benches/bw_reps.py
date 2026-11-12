"""Block-width reps: story+code x static w8/w11/w15, 3 reps each (GPU1)."""
import os, sys, time, json
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

PROMPTS = {
    "code": open(os.path.expanduser("~/exl3_pipeline/bench_code.txt")).read().strip(),
    "story": open(os.path.expanduser("~/exl3_pipeline/conv_prose_story.txt")).read().strip(),
}

def run(kind, ndt):
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                    draft_model=draft_model, draft_cache=draft_cache,
                    num_draft_tokens=ndt, record_draft_stats=True)
    prompt = ("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
              f"<|im_start|>user\n{PROMPTS[kind]}<|im_end|>\n<|im_start|>assistant\n")
    input_ids = tokenizer.encode(prompt, encode_special_tokens=True)
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
    out, t0 = "", None
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
    acc = sum(s[2] + 1 for s in stats)
    return round(ntok / dt, 1), round(acc / max(1, n), 2)

report = {}
for kind in ("story", "code"):
    for ndt in (8, 11, 15):
        tps_l, com_l = [], []
        for rep in range(3):
            tps, com = run(kind, ndt)
            tps_l.append(tps); com_l.append(com)
        report[f"{kind}_w{ndt}"] = dict(tps=tps_l, committed=com_l,
                                        mean=round(sum(tps_l) / 3, 1))
        print(f"[{kind}_w{ndt}] mean={report[f'{kind}_w{ndt}']['mean']} reps={tps_l} committed={com_l}", flush=True)
json.dump(report, open(os.path.expanduser("~/exl3_pipeline/bw_reps.json"), "w"), indent=2)
print("DONE")
