#!/usr/bin/env python3
"""Bug 5 repro: omp-style tool-loop history vs live server.

Shapes an OpenAI tool-loop conversation exactly as an agent client replays it
(assistant turns carry tool_calls with null/"" content, tool results as
role:"tool") and measures completion_tokens / finish_reason for thinking on/off,
several temperatures and history lengths.
"""
import json, sys, time, urllib.request

URL = "http://192.168.1.105:8900/v1/chat/completions"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]

CITIES = [("Tokyo", '{"temp_c": 18, "condition": "cloudy", "humidity": 62}'),
          ("Paris", '{"temp_c": 11, "condition": "rain", "humidity": 88}'),
          ("Oslo", '{"temp_c": -2, "condition": "snow", "humidity": 71}'),
          ("Cairo", '{"temp_c": 29, "condition": "sunny", "humidity": 24}'),
          ("Lima", '{"temp_c": 22, "condition": "overcast", "humidity": 80}'),
          ("Delhi", '{"temp_c": 34, "condition": "haze", "humidity": 58}'),
          ("Bogota", '{"temp_c": 14, "condition": "drizzle", "humidity": 92}'),
          ("Suva", '{"temp_c": 27, "condition": "humid", "humidity": 95}')]


def build_history(rounds, assistant_content, think_text=False):
    msgs = [{"role": "system",
             "content": "You are a helpful travel agent. Use the get_weather tool "
                        "for every city the user asks about, then summarize."}]
    for i in range(rounds):
        city, result = CITIES[i % len(CITIES)]
        msgs.append({"role": "user",
                     "content": f"What's the weather in {city} right now? (request {i+1})"})
        if think_text:
            content = (f"<think>The user asks about {city}. I should call get_weather."
                       f"</think>\nLet me check.") if assistant_content is not None else None
        else:
            content = assistant_content
        msgs.append({"role": "assistant", "content": content,
                     "tool_calls": [{"id": f"call_{i}", "type": "function",
                                     "function": {"name": "get_weather",
                                                  "arguments": json.dumps({"city": city})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "content": result})
    msgs.append({"role": "user",
                 "content": "Summarize all the weather results in one short paragraph."})
    return msgs


def call(msgs, **kw):
    body = {"model": "qwen3.8-27b-exl3-4bpw-prose", "messages": msgs,
            "max_tokens": 512, "tools": TOOLS, "stream": False}
    body.update(kw)
    t0 = time.time()
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"http": e.code, "error": e.read().decode()[:200]}
    ch = d["choices"][0]
    m = ch["message"]
    return {"http": 200, "finish": ch["finish_reason"],
            "tokens": d.get("usage", {}).get("completion_tokens"),
            "content_len": len(m.get("content") or ""),
            "tool_calls": len(m.get("tool_calls") or []),
            "content_head": (m.get("content") or "")[:80],
            "wall": round(time.time() - t0, 1)}


def main():
    results = []
    cases = []
    for rounds in (1, 2, 4, 8):
        for ac, tag in ((None, "null"), ("", "empty")):
            cases.append(dict(rounds=rounds, ac=ac, tag=tag,
                              think=True, temp=0.0, extra={}))
    # controls: thinking off, high temp, min_tokens, think-laden history
    cases.append(dict(rounds=4, ac=None, tag="null", think=False, temp=0.0, extra={}))
    cases.append(dict(rounds=4, ac=None, tag="null", think=False, temp=0.8, extra={}))
    cases.append(dict(rounds=4, ac=None, tag="null", think=True, temp=2.0, extra={}))
    cases.append(dict(rounds=4, ac=None, tag="null", think=True, temp=0.0,
                      extra={"min_tokens": 50}))
    cases.append(dict(rounds=8, ac="", tag="empty", think=True, temp=0.0,
                      extra={"chat_template_kwargs": {"enable_thinking": True},
                             "preserve_thinking": True}))

    for c in cases:
        msgs = build_history(c["rounds"], c["ac"])
        kw = {"enable_thinking": c["think"], "temperature": c["temp"]}
        kw.update(c["extra"])
        r = call(msgs, **kw)
        row = {k: c[k] for k in ("rounds", "tag", "think", "temp")} | {"extra": c["extra"]} | r
        results.append(row)
        print(json.dumps(row), flush=True)

    with open("repro_bug5_results.json", "w") as f:
        json.dump(results, f, indent=1)
    empty = [r for r in results if r.get("http") == 200 and r["tokens"] == 0]
    print(f"\nEMPTY (0 tokens): {len(empty)}/{len(results)}")
    for r in empty:
        print("  ", {k: r[k] for k in ("rounds", "tag", "think", "temp", "finish")})


if __name__ == "__main__":
    main()
