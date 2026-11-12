"""Blocker-1 probe: capture-first + state-mutation probe.
Answers: does the captured graph write the LIVE conv ring / GDN state rows?"""
import os, sys, time
import torch
sys.path.insert(0, os.path.expanduser("~/exl3_pipeline/exllamav3"))
os.environ["DSPINE_GRAPH"] = "1"
os.environ["DSPINE_VERIFY_GRAPH"] = "1"
os.environ["DSPINE_VERIFY_GRAPH_DEBUG"] = "1"
os.environ["EXL3_GDN_FORCE_TORCH"] = "1"
os.environ["EXL3_GRAPH_SAFE"] = "1"
os.environ["EXL3_BC_ATTN"] = "0"
os.environ["DSPINE_VG_TELLERY"] = "1"
os.environ["DSPINE_VG_SELFCHECK"] = "1"
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

# --- state probe: wrap first GDN module to remember its recurrent state layer ---
GDN_TYPE = None
gdn_mod = None
for m in model.modules:
    if type(m).__name__ == "TransformerBlock":
        at = getattr(m, "attn", None)
        if at is not None and "GatedDelta" in type(at).__name__:
            GDN_TYPE = type(at).__name__
            gdn_mod = at
            break
print(f"[probe] GDN module type: {GDN_TYPE}")

_orig_gdn = gdn_mod.forward
def _gdn_fwd(x, params, *a, **k):
    rsg = params.get("recurrent_states")
    y = _orig_gdn(x, params, *a, **k)
    if rsg:
        try:
            rsl = rsg[0].cache.get_recurrent_layer((gdn_mod.layer_idx, params.get("layer_instance", 0)))
            gdn_mod._last_rsl = rsl
        except Exception as e:
            pass
    return y
gdn_mod.forward = _gdn_fwd

def _h(t):
    return round(float(t.float().abs().sum().item()), 3) if t is not None else None

def _state_sig():
    rsl = getattr(gdn_mod, "_last_rsl", None)
    if rsl is None:
        return None
    return (_h(rsl.conv_state), _h(rsl.recurrent_state[0:1]), _h(rsl.recurrent_state[1:2]))

# --- wrap generator replay calls via subclass-free monkeypatch on iterate ---
code = open(os.path.expanduser("~/exl3_pipeline/bench_code.txt")).read().strip()
gen = Generator(model=model, cache=cache, tokenizer=tokenizer, draft_model=draft_model,
                draft_cache=draft_cache, num_draft_tokens=15, record_draft_stats=True)

_orig_replay_attr = Generator.__dict__.get("iterate")

# simplest: hook into graph capture by patching torch.cuda.CUDAGraph.replay
_orig_replay = torch.cuda.CUDAGraph.replay
def _replay(self):
    sig0 = _state_sig()
    _orig_replay(self)
    torch.cuda.synchronize()
    sig1 = _state_sig()
    print(f"[SPROBE] replay: state {sig0} -> {sig1}", flush = True)
torch.cuda.CUDAGraph.replay = _replay

prompt = ("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
          f"<|im_start|>user\n{code}<|im_end|>\n<|im_start|>assistant\n")
input_ids = tokenizer.encode(prompt, encode_special_tokens=True)
job = Job(input_ids=input_ids, max_new_tokens=120, decode_special_tokens=True, temperature=0.0,
          stop_conditions=[tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")])
gen.enqueue(job)
try:
    while gen.num_remaining_jobs():
        gen.iterate()
    print("COMPLETED_CLEAN")
except Exception as e:
    import traceback
    print(f"DIED: {type(e).__name__}: {e}")
    traceback.print_exc()
print("DONE")
