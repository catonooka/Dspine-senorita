"""Acceptance campaign (GPU1): think vs no-think x ndt 7/15, story + code, greedy 220 tok."""
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

def build_prompt(kind):
    if kind == "story":
        user = open(os.path.expanduser("~/exl3_pipeline/conv_prose_story.txt")).read().strip()
    else:
        user = open(os.path.expanduser("~/exl3_pipeline/bench_prompt.txt")).read()
    return ("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")

report = {}
for kind in ("story",):
    for think in (False, True):
        bans = None
        for ndt in (7,):
            generator = Generator(model = model, cache = cache, tokenizer = tokenizer,
                                  draft_model = draft_model, draft_cache = draft_cache,
                                  num_draft_tokens = ndt, record_draft_stats = True)
            prompt = build_prompt(kind)
            if not think:
                prompt = prompt + "<think>\n\n</think>\n\n"
            input_ids = tokenizer.encode(prompt, encode_special_tokens = True)
            job = Job(input_ids = input_ids, max_new_tokens = 220, decode_special_tokens = True,
                      temperature = 0.0, banned_strings = bans,
                      stop_conditions = [tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")])
            generator.enqueue(job)
            out = ""
            t0 = None
            while generator.num_remaining_jobs():
                results = generator.iterate()
                if t0 is None and out:
                    t0 = time.time()
                for r in results:
                    out += r.get("text", "")
            if t0 is None:
                t0 = time.time() - 1.0
            dt = max(1e-6, time.time() - t0)
            ntok = len(tokenizer.encode(out, encode_special_tokens = True).flatten())
            stats = job.draft_stats or []
            n = len(stats)
            acc_tok = sum(a for _, _, a in stats)
            drafted = ndt * n
            key = f"{kind}_{'think' if think else 'nothink'}_ndt{ndt}"
            report[key] = dict(
                tokens = ntok, rounds = n,
                committed_per_round = round(acc_tok / max(1, n) + 1, 2),
                accept_pct = round(100 * acc_tok / max(1, drafted), 2),
                tps = round(ntok / dt, 1),
            )
            print(f"[{key}] {json.dumps(report[key])}", flush = True)

json.dump(report, open(os.path.expanduser("~/exl3_pipeline/accept_campaign.json"), "w"), indent = 2)
print("DONE")
