"""Blocker-1 investigation (GPU2): find the remaining capture-illegal op with
CUDA_LAUNCH_BLOCKING=1 (errors surface at the offending launch), then check
whether an aborted capture corrupts the following eager rounds."""
import os, sys, time
import torch
sys.path.insert(0, os.path.expanduser("~/exl3_pipeline/exllamav3"))
os.environ["DSPINE_GRAPH"] = "1"
os.environ["DSPINE_VERIFY_GRAPH"] = "1"
os.environ["DSPINE_VERIFY_GRAPH_DEBUG"] = "1"
os.environ["EXL3_GDN_FORCE_TORCH"] = "1"
os.environ["EXL3_GRAPH_SAFE"] = "1"
os.environ["EXL3_BC_ATTN"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["DSPINE_VG_BPROBE"] = "1"
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

code = open(os.path.expanduser("~/exl3_pipeline/bench_code.txt")).read().strip()
gen = Generator(model=model, cache=cache, tokenizer=tokenizer, draft_model=draft_model,
                draft_cache=draft_cache, num_draft_tokens=15, record_draft_stats=True)
prompt = ("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
          f"<|im_start|>user\n{code}<|im_end|>\n<|im_start|>assistant\n")
input_ids = tokenizer.encode(prompt, encode_special_tokens=True)
job = Job(input_ids=input_ids, max_new_tokens=120, decode_special_tokens=True, temperature=0.0,
          stop_conditions=[tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")])
gen.enqueue(job)
out = ""
try:
    while gen.num_remaining_jobs():
        results = gen.iterate()
        for r in results:
            out += r.get("text", "")
    print("COMPLETED_CLEAN")
except Exception as e:
    import traceback
    print(f"DIED: {type(e).__name__}: {e}")
    traceback.print_exc()
    # corruption check: can new jobs still run after the failure?
    try:
        j2 = Job(input_ids=input_ids, max_new_tokens=8, decode_special_tokens=True,
                 temperature=0.0,
                 stop_conditions=[tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")])
        gen2 = Generator(model=model, cache=cache, tokenizer=tokenizer,
                         draft_model=draft_model, draft_cache=draft_cache,
                         num_draft_tokens=15)
        gen2.enqueue(j2)
        o2 = ""
        while gen2.num_remaining_jobs():
            for r in gen2.iterate():
                o2 += r.get("text", "")
        print(f"POST_FAIL_RECOVERY_OK len={len(o2)}")
    except Exception as e2:
        print(f"POST_FAIL_ALSO_DEAD: {type(e2).__name__}: {e2}")
print("DONE")
