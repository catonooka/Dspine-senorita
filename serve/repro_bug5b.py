#!/usr/bin/env python3
"""Bug 5 repro, round 2: faithful mid-loop shapes.

All cases END the history where an agent loop actually pauses for the next
completion: on role:"tool" results (server renders them as plain user turns).
Variants: empty tool results, think-laden assistant history, parallel calls,
plain assistant answers interleaved.
"""
import json, time, urllib.request

URL = "http://192.168.1.105:8900/v1/chat/completions"
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
CITIES = [("Tokyo", '{"temp_c": 18, "condition": "cloudy", "humidity": 62}'),
          ("Paris", '{"temp_c": 11, "condition": "rain", "humidity": 88}'),
          ("Oslo", '{"temp_c": -2, "condition": "snow", "humidity": 71}'),
          ("Cairo", '{"temp_c": 29, "condition": "sunny", "humidity": 24}')]


def tool_round(msgs, i, city, result, ac=None, parallel=False):
    msgs.append({"role": "user", "content": f"Weather in {city}? (req {i+1})"})
    calls = [{"id": f"call_{i}", "type": "function",
              "function": {"name": "get_weather", "arguments": json.dumps({"city": city})}}]
    if parallel:
        c2, r2 = CITIES[(i + 1) % 4]
        calls.append({"id": f"call_{i}b", "type": "function",
                      "function": {"name": "get_weather", "arguments": json.dumps({"city": c2})}})
    msgs.append({"role": "assistant", "content": ac, "tool_calls": calls})
    msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "content": result})
    if parallel:
        c2, r2 = CITIES[(i + 1) % 4]
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}b", "content": r2})


def history(rounds=4, ac=None, empty_tool=False, think_hist=False,
            plain_answers=False, parallel=False):
    msgs = [{"role": "system",
             "content": "You are a helpful agent. Use get_weather for each city asked "
                        "about. After collecting results, answer concisely."}]
    msgs.append({"role": "user", "content": "I'm planning a trip. Check the weather in "
                 + ", ".join(c for c, _ in CITIES[:rounds]) + " and advise."})
    for i in range(rounds):
        city, result = CITIES[i % 4]
        tool_round(msgs, i, city, "" if empty_tool else result, ac=ac, parallel=parallel)
        if plain_answers and i < rounds - 1:
            msgs.append({"role": "user", "content": f"Anything else about {city}?"})
            msgs.append({"role": "assistant",
                         "content": f"{city} looks fine for sightseeing."})
        if think_hist and i < rounds - 1:
            msgs.append({"role": "user", "content": f"Notes for {city}?"})
            msgs.append({"role": "assistant",
                         "content": f"<think>User wants notes for {city}. Provide a "
                                    f"one-liner.</think>\n{city}: pack accordingly."})
    return msgs


def call(msgs, **kw):
    body = {"model": "qwen3.8-27b-exl3-4bpw-prose", "messages": msgs,
            "max_tokens": 512, "tools": TOOLS, "stream": False}
    body.update(kw)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"http": e.code, "err": e.read().decode()[:150]}
    ch = d["choices"][0]
    m = ch["message"]
    return {"http": 200, "finish": ch["finish_reason"],
            "tokens": d.get("usage", {}).get("completion_tokens"),
            "clen": len(m.get("content") or ""),
            "tc": len(m.get("tool_calls") or []),
            "head": (m.get("content") or "")[:60], "wall": round(time.time() - t0, 1)}


CASES = [
    ("end-at-tool r4 null", dict(rounds=4, ac=None)),
    ("end-at-tool r4 empty", dict(rounds=4, ac="")),
    ("end-at-tool r2 null", dict(rounds=2, ac=None)),
    ("end-at-tool r4 empty-tool-result", dict(rounds=4, ac=None, empty_tool=True)),
    ("end-at-tool r4 think-hist", dict(rounds=4, ac=None, think_hist=True)),
    ("end-at-tool r4 plain-answers", dict(rounds=4, ac=None, plain_answers=True)),
    ("end-at-tool r4 parallel", dict(rounds=4, ac=None, parallel=True)),
    ("end-at-tool r4 null temp2", dict(rounds=4, ac=None)),
]

out = []
for name, kw in CASES:
    temp = 2.0 if "temp2" in name else 0.0
    r = call(history(**kw), enable_thinking=True, temperature=temp)
    row = {"case": name, "temp": temp} | r
    out.append(row)
    print(json.dumps(row), flush=True)

empty = [r for r in out if r.get("tokens") == 0]
print(f"\nEMPTY: {len(empty)}/{len(out)}")
