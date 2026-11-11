"""Round anatomy: exact ms split of every spec round (draft / gen / gap), prose vs code."""
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

def build_prompt(kind, think):
    if kind == "prose":
        user = open(os.path.expanduser("~/exl3_pipeline/conv_prose_story.txt")).read().strip()
    else:
        user = open(os.path.expanduser("~/exl3_pipeline/bench_prompt.txt")).read()
    tail = ""    # no manual soft-switch: instant-stop bug; thinking tokens included in window
    return ("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{tail}")

report = {}
for kind, think in (("code", True), ("prose", True)):
    generator = Generator(model = model, cache = cache, tokenizer = tokenizer,
                          draft_model = draft_model, draft_cache = draft_cache,
                          num_draft_tokens = 15, record_draft_stats = True)
    prompt = build_prompt(kind, think)
    input_ids = tokenizer.encode(prompt, encode_special_tokens = True)
    job = Job(input_ids = input_ids, max_new_tokens = 220, decode_special_tokens = True,
              temperature = 0.0,
              stop_conditions = [tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")])
    generator.enqueue(job)
    generator._rt_on = None; generator._rt_draft = []; generator._rt_gen = []
    t0 = None
    while generator.num_remaining_jobs():
        results = generator.iterate()
        if t0 is None and generator._rt_draft:
            t0 = time.time()
        for r in results:
            pass
    dt = time.time() - t0 if t0 else -1
    d = generator._rt_draft; g = generator._rt_gen
    n = min(len(d), len(g))
    stats = job.draft_stats or []
    committed = sum(a for _, _, a in stats) / max(1, len(stats))
    gap = (sum(g[:n]) - 0)  # gen includes verify+walk+commit; gap = total-draft-gen measured below
    total = dt / max(1, n)
    md = sum(d[:n]) / n * 1000
    mg = sum(g[:n]) / n * 1000
    key = f"{kind}{'_think' if think else ''}"
    report[key] = dict(
        rounds = n, committed_per_round = round(committed + 1, 2),
        round_ms = round(total * 1000, 1), draft_ms = round(md, 1), gen_ms = round(mg, 1),
        gap_ms = round(total * 1000 - md - mg, 1),
        tps = round((committed + 1) / total, 1),
    )
    print(f"[{key}] {json.dumps(report[key])}", flush = True)

json.dump(report, open(os.path.expanduser("~/exl3_pipeline/round_anatomy.json"), "w"), indent = 2)
print("DONE")
