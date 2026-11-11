"""SpineCamp 2: push code (conf up w/ high acceptance) + sparse head A/B on best variants."""
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

def run(tag, kind, dynamic, conf, hybrid, sparse):
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                    draft_model=draft_model, draft_cache=draft_cache,
                    num_draft_tokens=15, record_draft_stats=True,
                    dynamic_draft_tokens=dynamic, draft_confidence=conf,
                    hybrid_draft_tokens=hybrid, sparse_draft_head=sparse)
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
    dt = max(1e-6, time.time() - (t0 if t0 is not None else time.time()))
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
# (tag, kind, dynamic, conf, hybrid, sparse)
SP = 32768
VARIANTS = [
    ("code_dyn_c035", "code", True, 0.35, 0, 0),
    ("code_dyn_c035_sp", "code", True, 0.35, 0, SP),
    ("code_dyn_c05", "code", True, 0.5, 0, 0),
    ("code_dyn_c05_sp", "code", True, 0.5, 0, SP),
    ("code_dyn_c025_sp", "code", True, 0.25, 0, SP),
    ("story_stat_hyb8", "story", False, 0.25, 8, 0),
    ("story_stat_hyb8_sp", "story", False, 0.25, 8, SP),
    ("story_dyn_c025_sp", "story", True, 0.25, 0, SP),
]
for tag, kind, dyn, conf, hyb, sp in VARIANTS:
    report[tag] = run(tag, kind, dyn, conf, hyb, sp)
json.dump(report, open(os.path.expanduser("~/exl3_pipeline/spinecamp2.json"), "w"), indent=2)
print("DONE")
