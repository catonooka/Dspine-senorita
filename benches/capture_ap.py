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
os.environ["DSPINE_SHAPE_DEBUG"] = "1"
os.environ["DSPINE_VG_TELLERY"] = "1"
os.environ["DSPINE_VG_SELFCHECK"] = "1"
os.environ["DSPINE_VG_LAYERDIFF"] = "1"
os.environ["DSPINE_VG_ATTNPROBE"] = "1"
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

def _install_sprobe(gen, model):
    def _hash(t):
        return float(t.float().sum().item()) if t is not None else None
    def _sprobe(g, key):
        # find first GDN recurrent layer via a GDN module
        import torch
        gdn = None
        for m in model.modules:
            if type(m).__name__ == "GatedDeltaNet":
                gdn = m; break
        rs = None
        if gdn is not None:
            try:
                rs = gdn._last_rsl
            except AttributeError:
                pass
        import torch
        logits1 = gen._vg_out[key].clone()
        torch.cuda.synchronize()
        h0 = (_hash(rs.conv_state), _hash(rs.recurrent_state[0]), _hash(rs.recurrent_state[1])) if rs is not None else None
        g.replay()
        logits2 = gen._vg_out[key].clone()
        torch.cuda.synchronize()
        h1 = (_hash(rs.conv_state), _hash(rs.recurrent_state[0]), _hash(rs.recurrent_state[1])) if rs is not None else None
        d = (logits1.float() - logits2.float()).abs()
        print(f"[SPROBE] l1vsl2 max_abs={d.max().item():.4f} | ring {h0[0] if h0 else None}->{h1[0] if h1 else None} "
              f"row0 {h0[1] if h0 else None}->{h1[1] if h1 else None} row1 {h0[2] if h0 else None}->{h1[2] if h1 else None}",
              flush=True)
        return logits2
    gen._vg_sprobe = _sprobe

# hook: wrap one GDN module to capture its rsl (recurrent state layer) at forward time
_orig_gdn_fwd = None
def _patch_gdn(gen, model):
    global _orig_gdn_fwd
    for m in model.modules:
        if type(m).__name__ == "GatedDeltaNet":
            _orig = m.forward
            def _f(x, params, *a, **k):
                rsg = params.get("recurrent_states")
                if rsg and not hasattr(params.get("_sprobe_rsl", None), "conv_state"):
                    try:
                        rsl = rsg[0].cache.get_recurrent_layer((m.layer_idx, params.get("layer_instance", 0)))
                        params["_sprobe_rsl"] = rsl
                    except Exception:
                        pass
                y = _orig(x, params, *a, **k)
                rsl = params.get("_sprobe_rsl")
                if rsl is not None:
                    m._last_rsl = rsl
                return y
            m.forward = _f
            break

gen_ready = False
_orig_gen_iterate = None
def _wire():
    _install_sprobe(gen, model)
    _patch_gdn(gen, model)
model=model, cache=cache, tokenizer=tokenizer, draft_model=draft_model,
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
