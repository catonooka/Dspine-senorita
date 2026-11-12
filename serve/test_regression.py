#!/usr/bin/env python3
"""Live regression suite for the Bug-5 fix (instant-EOS with thinking on).

T1  normal completion, thinking off and on
T2  max_completion_tokens honored (finish length at cap)
T3  get_weather tool_call round-trip incl. tool-loop history (new template path)
T4  streaming delta.tool_calls
T5  reasoning_effort none == thinking-off alias
T6  BUG5 PROBE: "reply empty" instruction + thinking ON -> must NOT be 0 tokens
T7  min_tokens honored: "reply empty" + thinking off + min_tokens=50 -> >= 50
T8  temperature reaches the engine: temp 0 deterministic, high temp varies
T9  think_rescue intact (b163e11): cap mid-think -> closed think + answer
"""
import json, time, urllib.request

BASE = "http://192.168.1.105:8900"
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
RESULTS = []


def call(msgs, stream=False, **kw):
    body = {"model": "qwen3.8-27b-exl3-4bpw-prose", "messages": msgs,
            "max_tokens": 300, "stream": stream}
    body.update(kw)
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        if stream:
            chunks, tool_calls = [], []
            content = ""
            finish = None
            while True:
                line = r.readline()
                if not line:
                    break
                if line.startswith(b"data: "):
                    d = line[6:].strip()
                    if d == b"[DONE]":
                        break
                    try:
                        obj = json.loads(d)
                    except json.JSONDecodeError:
                        continue  # stats frame
                    ch = obj.get("choices") or [{}]
                    dl = ch[0].get("delta", {})
                    if dl.get("content"):
                        content += dl["content"]
                    for tc in dl.get("tool_calls") or []:
                        tool_calls.append(tc)
                    if ch[0].get("finish_reason"):
                        finish = ch[0]["finish_reason"]
            return {"content": content, "finish": finish, "tool_calls": tool_calls}
        d = json.loads(r.read())
    ch = d["choices"][0]
    m = ch["message"]
    return {"finish": ch["finish_reason"],
            "tokens": d.get("usage", {}).get("completion_tokens"),
            "content": m.get("content") or "",
            "tool_calls": m.get("tool_calls") or []}


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


# T1 normal completion
r = call([{"role": "user", "content": "Name the capital of France in one word."}],
         enable_thinking=False, temperature=0.0)
check("T1a no-think completion", "paris" in r["content"].lower() and r["tokens"] > 0,
      f"tok={r['tokens']} finish={r['finish']}")
r = call([{"role": "user", "content": "Name the capital of France in one word."}],
         enable_thinking=True, temperature=0.0)
check("T1b think completion", "paris" in r["content"].lower() and r["tokens"] > 0
      and r["content"].startswith("<think>"),
      f"tok={r['tokens']} finish={r['finish']} head={r['content'][:16]!r}")

# T2 max_completion_tokens
r = call([{"role": "user", "content": "Write a long story about a lighthouse."}],
         enable_thinking=False, temperature=0.0, max_completion_tokens=150)
check("T2 max_completion_tokens=150", r["finish"] == "length" and r["tokens"] >= 150,
      f"tok={r['tokens']} finish={r['finish']}")

# T3 tool_call round-trip incl. replayed tool-loop history (new template path)
r = call([{"role": "user", "content": "What's the weather in Tokyo? Use the tool."}],
         tools=TOOLS, enable_thinking=True, temperature=0.0)
ok = (r["finish"] == "tool_calls" and r["tool_calls"]
      and r["tool_calls"][0]["function"]["name"] == "get_weather"
      and "Tokyo" in r["tool_calls"][0]["function"]["arguments"])
check("T3a fresh tool_call", ok, f"finish={r['finish']}")
hist = [
    {"role": "system", "content": "You are a weather agent."},
    {"role": "user", "content": "Weather in Tokyo?"},
    {"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_1", "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 18, "condition": "cloudy"}'},
]
r = call(hist + [{"role": "user", "content": "Now Paris, same way."}],
         tools=TOOLS, enable_thinking=True, temperature=0.0)
ok = (r["finish"] == "tool_calls" and r["tool_calls"]
      and r["tool_calls"][0]["function"]["name"] == "get_weather")
check("T3b tool-loop history -> new tool_call", ok, f"finish={r['finish']}")
r = call(hist, tools=TOOLS, enable_thinking=True, temperature=0.0)
check("T3c end-at-tool history answers", r["tokens"] > 0 and len(r["content"]) > 20,
      f"tok={r['tokens']} finish={r['finish']}")

# T4 streaming delta.tool_calls
r = call([{"role": "user", "content": "Weather in Oslo? Use the tool."}],
         tools=TOOLS, enable_thinking=True, temperature=0.0, stream=True)
ok = r["finish"] == "tool_calls" and r["tool_calls"] and \
     r["tool_calls"][0]["function"]["name"] == "get_weather"
check("T4 stream delta.tool_calls", ok,
      f"finish={r['finish']} calls={len(r['tool_calls'])}")

# T5 reasoning_effort none == thinking off
r = call([{"role": "user", "content": "2+2? One word."}],
         reasoning_effort="none", temperature=0.0)
check("T5 reasoning_effort=none", not r["content"].startswith("<think>")
      and r["tokens"] > 0, f"head={r['content'][:16]!r}")

# T6 BUG5 PROBE: thinking ON + "reply empty" instruction -> non-zero tokens
P_EMPTY = [
    {"role": "system", "content": "You are a test dummy. When asked for an empty "
                                   "reply, output NOTHING at all."},
    {"role": "user", "content": "Reply with an empty message now. Output absolutely "
                                "nothing. No words, no punctuation, no spaces."}]
empties = 0
for _ in range(5):
    r = call(P_EMPTY, enable_thinking=True, temperature=0.0)
    if r["tokens"] == 0:
        empties += 1
check("T6 bug5 probe thinking-on never empty", empties == 0,
      f"empties={empties}/5")

# T7 min_tokens honored (thinking off, would-be-empty reply)
r = call(P_EMPTY, enable_thinking=False, temperature=0.0, min_tokens=1)
check("T7a min_tokens=1 blocks empty", r["tokens"] >= 1, f"tok={r['tokens']}")
r = call(P_EMPTY, enable_thinking=False, temperature=0.0, min_tokens=50)
check("T7b min_tokens=50 floor", r["tokens"] >= 50, f"tok={r['tokens']}")

# T8 temperature reaches engine
a = call([{"role": "user", "content": "Tell me a two-sentence bedtime story."}],
         enable_thinking=False, temperature=0.0)
b = call([{"role": "user", "content": "Tell me a two-sentence bedtime story."}],
         enable_thinking=False, temperature=0.0)
check("T8a temp 0 deterministic", a["content"] == b["content"])
cs = set()
for _ in range(4):
    cs.add(call([{"role": "user", "content": "Tell me a two-sentence bedtime story."}],
                enable_thinking=False, temperature=1.6, top_p=0.95)["content"])
check("T8b temp 1.6 varies", len(cs) > 1, f"distinct={len(cs)}/4")

# T9 think_rescue intact: cap hit mid-think -> closed think + answer
r = call([{"role": "user", "content": "Why is the sky blue? Explain thoroughly."},
         ], enable_thinking=True, temperature=0.0, max_tokens=8, think_rescue=True)
ok = "</think>" in r["content"] and r["tokens"] > 8
check("T9 think_rescue closes think", ok,
      f"tok={r['tokens']} finish={r['finish']} closed={'</think>' in r['content']}")

print()
bad = [n for n, ok in RESULTS if not ok]
print(f"{len(RESULTS) - len(bad)}/{len(RESULTS)} passed" + (f"; FAILED: {bad}" if bad else ""))
