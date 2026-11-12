import os, sys, time
import torch
sys.path.insert(0, os.path.expanduser("~/exl3_pipeline/exllamav3"))
os.environ["DSPINE_GRAPH"] = "1"
from exllamav3 import Config, Model, Cache, Tokenizer

drafter_dir = os.path.expanduser("~/exl3_pipeline/output_exl3/Drafter-DSpine-EXL3-5.0bpw-h8")
model_dir = os.path.expanduser("~/exl3_pipeline/output_exl3/Model-27B-EXL3-4.0bpw-h8")
config = Config.from_directory(model_dir)
model = Model.from_config(config)
dcfg = Config.from_directory(drafter_dir)
dm = Model.from_config(dcfg, component="text")
dcache = Cache(dm, max_num_tokens=32768, max_history=16, max_batch_size=1, recurrent_dtype=torch.half)
dm.load(progressbar=False, max_batch_size=1, autosplit_no_forward=True, max_chunk_size=2048)
model.load(draft_model=dm, progressbar=False, max_batch_size=1, autosplit_no_forward=True, max_chunk_size=2048)
if not dcache.initialized:
    dm._mark_caches_initialized()
torch.cuda.empty_cache()
dm.attach_to(model)

print("=== call 1 (build) ===")
t0 = time.perf_counter(); torch.cuda.synchronize()
dm._ensure_tables(model)
torch.cuda.synchronize()
print(f"call1: {time.perf_counter()-t0:.3f}s  _tg is None: {dm._tg is None}")
if dm._tg is not None:
    print("tg shape:", tuple(dm._tg.shape), "tsv shape:", tuple(dm._tsv.shape))
print("=== call 2 (should early-return) ===")
t0 = time.perf_counter()
dm._ensure_tables(model)
torch.cuda.synchronize()
print(f"call2: {(time.perf_counter()-t0)*1000:.2f}ms")
print("=== call 3 ===")
t0 = time.perf_counter()
dm._ensure_tables(model)
torch.cuda.synchronize()
print(f"call3: {(time.perf_counter()-t0)*1000:.2f}ms")
print("MICROTEST_DONE")
