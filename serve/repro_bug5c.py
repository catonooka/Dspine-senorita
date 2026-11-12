#!/usr/bin/env python3
"""Bug 5 repro, round 3: LONG context (~55K tokens) agent session, ends at tool.

Realistic omp-shaped history: many tools, assistant turns with null content +
native tool_calls, role:tool results containing file contents / shell dumps.
Control case: consecutive assistant turns (guaranteed nothing-to-add shape).
"""
import json, time, urllib.request

URL = "http://192.168.1.105:8900/v1/chat/completions"
TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                    "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a file",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search file contents",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Apply an edit",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                    "old": {"type": "string"}, "new": {"type": "string"}},
                    "required": ["path", "old", "new"]}}}]

CODE = '''import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job

def bench(lane, window, dynamic, conf, prompt_len=400):
    """Bench one lane: story + code prompts, 3 reps, return mean t/s."""
    config = Config.from_directory(MODEL_DIR)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=90112, max_history=16)
    drafter = Model.from_config(DRAFTER_DIR, component="text")
    dcache = Cache(drafter, max_num_tokens=90112, max_history=16)
    drafter.load(progressbar=False)
    model.load(draft_model=drafter, progressbar=False)
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                    draft_model=drafter, draft_cache=dcache,
                    num_draft_tokens=window, dynamic_draft_tokens=dynamic,
                    draft_confidence=conf)
    results = []
    for rep in range(3):
        ids = tokenizer.encode(PROMPTS[prompt_len], encode_special_tokens=True)
        job = Job(input_ids=ids, max_new_tokens=400, decode_special_tokens=True,
                  stop_conditions=[tokenizer.eos_token_id])
        gen.enqueue(job)
        out, t0 = "", None
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                d = r.get("text", "")
                if d:
                    if t0 is None: t0 = time.time()
                    out += d
        results.append(len(out.split()) / (time.time() - t0))
    return sum(results) / len(results)
''' + "# padding comment line to give the file realistic agent-session bulk\n" * 40

SYS = ("You are a coding agent working in a repository. Use the bash, read_file, "
       "grep and edit tools to investigate and modify the codebase. Be systematic: "
       "read before editing, verify after editing. Report concisely.")


def long_history(rounds=20):
    msgs = [{"role": "system", "content": SYS}]
    msgs.append({"role": "user", "content":
                 "Profile the benchmark script and figure out why lane dspine is "
                 "slower than dflash2. Then write up findings."})
    for i in range(rounds):
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"call_{i}", "type": "function",
                                     "function": {
                                         "name": "read_file" if i % 3 == 0 else "bash",
                                         "arguments": json.dumps({
                                             "path": f"bench_{i}.py",
                                             "command": f"python bench_{i}.py --rep 3"})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}",
                     "content": f"[file bench_{i}.py | lines 1-{CODE.count(chr(10))+1}]\n{CODE}"
                                if i % 3 == 0 else
                                f"$ python bench_{i}.py --rep 3\nrep1 t/s=57.5 "
                                f"committed=2.60\nrep2 t/s=54.8 committed=2.06\nrep3 "
                                f"t/s=60.2 committed=2.78\nround_ms=46.2 draft_ms=9.0 "
                                f"gen_ms=37.5 width=15\n" + "log line: verify pass "
                                f"completed ok, no truncation\n" * 25})
    return msgs


def call(msgs, **kw):
    body = {"model": "qwen3.8-27b-exl3-4bpw-prose", "messages": msgs,
            "max_tokens": 400, "tools": TOOLS, "stream": False}
    body.update(kw)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"http": e.code, "err": e.read().decode()[:150]}
    ch = d["choices"][0]
    m = ch["message"]
    return {"http": 200, "finish": ch["finish_reason"],
            "tokens": d.get("usage", {}).get("completion_tokens"),
            "clen": len(m.get("content") or ""),
            "tc": len(m.get("tool_calls") or []),
            "head": (m.get("content") or "")[:50], "wall": round(time.time() - t0, 1)}


def main():
    hist = long_history(20)
    nchars = sum(len(json.dumps(m)) for m in hist)
    print(f"history: {len(hist)} msgs, ~{nchars} chars (~{nchars//4} tok est)", flush=True)

    # Control: consecutive assistant turns, short (guaranteed nothing-to-add shape)
    ctrl = [{"role": "system", "content": SYS},
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "Thanks, that's all I needed."},
            {"role": "assistant", "content": "You're welcome! Goodbye."},
            {"role": "user", "content": "Bye."}]
    for think in (True, False):
        r = call(ctrl, enable_thinking=think, temperature=0.0)
        print(json.dumps({"case": "ctrl-consec-assist", "think": think} | r), flush=True)

    for name, kw in [
            ("long think-on t0", {"enable_thinking": True, "temperature": 0.0}),
            ("long think-on t0.8", {"enable_thinking": True, "temperature": 0.8}),
            ("long think-off t0", {"enable_thinking": False, "temperature": 0.0}),
            ("long think-on min50", {"enable_thinking": True, "temperature": 0.0,
                                     "min_tokens": 50})]:
        r = call(hist, **kw)
        print(json.dumps({"case": name} | kw | r), flush=True)


if __name__ == "__main__":
    main()
