"""Tree lossless probe: linear-chain tree vs chain engine, byte-compare (GPU1)."""
import os, sys, time, json
import torch
sys.path.insert(0, os.path.expanduser("~/exl3_pipeline/exllamav3"))
os.environ.setdefault("DSPINE_GRAPH", "1")
os.environ["DSPINE_TREE_LINEAR"] = "1"
from exllamav3 import Config, Model, Cache, Tokenizer, Job
from exllamav3.generator.tree_engine import TreeGenerator

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

generator = TreeGenerator(model = model, cache = cache, tokenizer = tokenizer,
                          draft_model = draft_model, draft_cache = draft_cache,
                          num_draft_tokens = 15, tree_cap = 31, record_draft_stats = True)

user = open(os.path.expanduser("~/exl3_pipeline/conv_prose_story.txt")).read().strip()
prompt = ("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
          f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
input_ids = tokenizer.encode(prompt, encode_special_tokens = True)

texts = {}
for mode in ("chain", "linear_tree"):
    generator.tree_cap = 0 if mode == "chain" else 31
    os.environ["DSPINE_TREE_LINEAR"] = "0" if mode == "chain" else "1"
    job = Job(input_ids = input_ids, max_new_tokens = 220, decode_special_tokens = True,
              temperature = 0.0,
              stop_conditions = [tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")])
    generator.enqueue(job)
    out = ""
    import traceback
    while generator.num_remaining_jobs():
        try:
            results = generator.iterate()
        except Exception:
            traceback.print_exc(); break
        for r in results:
            out += r.get("text", "")
    texts[mode] = out
    if mode == "chain" and os.environ.get("DSPINE_TREE_CAPTURE"):
        from exllamav3.generator.generator import TREE_CAPTURE
        if TREE_CAPTURE:
            c = TREE_CAPTURE[0]
            print(f"[chainR1] window={[int(c['draft'][0, i]) for i in range(min(5, c['draft'].shape[-1]))]}"
                  f" anchor_arg={int(c['target_argmax'][0, 0])}", flush = True)
    print(f"[{mode}] len={len(out)} {out[:120]!r}", flush = True)

t_t = tokenizer.encode(texts["linear_tree"], encode_special_tokens = True).flatten()
c_t = tokenizer.encode(texts["chain"], encode_special_tokens = True).flatten()
ident = 0
for i in range(min(len(t_t), len(c_t))):
    if t_t[i] != c_t[i]:
        break
    ident = i + 1
print(f"[PROBE] linear_tree={len(t_t)} tok, chain={len(c_t)} tok, identical_prefix={ident}, "
      f"LOSSLESS={ident == min(len(t_t), len(c_t))}", flush = True)
if ident < min(len(t_t), len(c_t)):
    print(f"[PROBE] diverge: tree={texts['linear_tree'][ident*3:ident*3+80]!r}", flush = True)
    print(f"[PROBE]          chain={texts['chain'][ident*3:ident*3+80]!r}", flush = True)
print("DONE")
