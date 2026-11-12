#!/usr/bin/env python3
"""Byte-equivalence test: serve_prose.chat_prompt vs official chat_template.jinja.

Renders both for a battery of message shapes (tools on/off, tool_calls
single/parallel, null/empty/text content, consecutive tool results, thinking
on/off, all efforts) and diffs. Run with the exl3-env python (jinja2 present).
"""
import json, sys
sys.path.insert(0, "/home/aisever/Dspine-senorita/serve")
sys.path.insert(0, "/home/aisever/exl3_pipeline/exllamav3")

from jinja2 import Environment, StrictUndefined
import serve_prose as sp

_src = open("/home/aisever/exl3_pipeline/raw_base/chat_template.jinja").read()
_src = _src.replace("{%- generation %}", "").replace("{%- endgeneration %}", "")
tpl = Environment(undefined=StrictUndefined).from_string(_src)

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]

def tc(name, args):
    if isinstance(args, str):
        args = args
    return {"id": "call_x", "type": "function",
            "function": {"name": name, "arguments": args}}

CASES = {
    "plain": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Weather in Tokyo?"}],
    "tools-single-null": [
        {"role": "system", "content": "Agent."},
        {"role": "user", "content": "Weather in Tokyo?"},
        {"role": "assistant", "content": None, "tool_calls": [
            tc("get_weather", {"city": "Tokyo"})]},
        {"role": "tool", "tool_call_id": "call_x",
         "content": '{"temp_c": 18}'},
        {"role": "user", "content": "Thanks, summarize."}],
    "tools-parallel-empty": [
        {"role": "user", "content": "Tokyo and Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [
            tc("get_weather", {"city": "Tokyo"}),
            tc("get_weather", '{"city": "Paris"}')]},
        {"role": "tool", "tool_call_id": "call_x", "content": "18c"},
        {"role": "tool", "tool_call_id": "call_y", "content": "11c"}],
    "tools-with-text": [
        {"role": "user", "content": "Check Oslo."},
        {"role": "assistant", "content": "Sure, checking now.", "tool_calls": [
            tc("get_weather", {"city": "Oslo", "units": "metric"})]},
        {"role": "tool", "tool_call_id": "call_x", "content": "-2c snow"}],
    "think-in-history": [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "<think>\nmath\n</think>\n\nFour."},
        {"role": "user", "content": "And 2+3?"}],
    "end-at-tool": [
        {"role": "user", "content": "Tokyo?"},
        {"role": "assistant", "content": None, "tool_calls": [tc("get_weather", {"city": "Tokyo"})]},
        {"role": "tool", "tool_call_id": "call_x", "content": "18c"}],
}

fails = 0
for name, msgs in CASES.items():
    for tools in (None, TOOLS):
        for think in (True, False):
            for eff in ("xhigh", "medium", "low"):
                ref = tpl.render(messages=msgs, tools=tools, add_generation_prompt=True,
                                 enable_thinking=think, reasoning_effort=eff)
                mine = sp.chat_prompt(msgs, enable_thinking=think,
                                      reasoning_effort=eff, tools=tools)
                tag = f"{name}/tools={bool(tools)}/think={think}/{eff}"
                if ref != mine:
                    fails += 1
                    print(f"DIFF {tag}")
                    for i, (a, b) in enumerate(zip(ref, mine)):
                        if a != b:
                            print("  ref :", repr(ref[max(0,i-40):i+60]))
                            print("  mine:", repr(mine[max(0,i-40):i+60]))
                            break
                    if len(ref) != len(mine):
                        print("  len ref", len(ref), "mine", len(mine))
                        print("  ref tail :", repr(ref[-90:]))
                        print("  mine tail:", repr(mine[-90:]))
print("ALL MATCH" if fails == 0 else f"{fails} MISMATCHES")
