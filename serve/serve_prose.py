#!/usr/bin/env python
"""serve_prose.py — max-prose-speed server for Qwen3.8-27B EXL3 4.0bpw-h8 (1x RTX 3090).

OpenAI-compatible (stdlib only):
  POST /v1/chat/completions   {"messages": [...], "temperature": 0, "max_tokens": N, "stream": bool}
  GET  /v1/models

Champion config (measured 2026-11-12, see REPORT.md):
  drafter DFlash2-EXL3-5.0bpw-h8, window 7, dynamic verify, draft_confidence 0.25
  -> 64.7 t/s think / 67.0 t/s no-think @400-tok story bench (single stream)

Alternate lanes:
  --drafter dspine      DSpine BF16, window 15, c0.25   (its best prose policy)
  --drafter dspine-exl3 DSpine EXL3, window 15, c0.25
  --drafter dflash2-bf16 DFlash2 BF16, window 7, c0.25
  --no-dynamic          static window (acceptance-chasing, higher per-round cost)

Usage:
  CUDA_VISIBLE_DEVICES=0 python serve_prose.py --port 8900
"""
import argparse, json, os, re, sys, time, threading, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("DSPINE_GRAPH", "1")
sys.path.insert(0, os.path.expanduser("~/exl3_pipeline/exllamav3"))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ComboSampler

VISION_ENABLED = True
try:
    from PIL import Image
    import base64, io, urllib.request
except ImportError:
    VISION_ENABLED = False

LANES = {
    # lane: (drafter dir, window, context pool tokens) — pool sized so weights+pool fit 24G
    "dspine":       ("~/exl3_pipeline/output_exl3/Drafter-DSpine-EXL3-5.0bpw-h8", 15, 90112),
    "dspine-bf16":  ("~/exl3_pipeline/raw_drafter", 15, 65536),
    "dflash2":      ("~/exl3_pipeline/output_exl3/Drafter-DFlash2-EXL3-5.0bpw-h8", 7, 122880),
    "dflash2-bf16": ("~/exl3_pipeline/raw_dflash2", 7, 98304),
}
MODEL_DIR = "~/exl3_pipeline/output_exl3/Model-27B-EXL3-4.0bpw-h8"


def build_generator(lane, dynamic, conf, window=None, kv8=False, pool_override=None, hdt=0):
    drafter_dir, lane_window, pool = LANES[lane]
    window = window or lane_window  # --window override (bw_reps per-class best)
    config = Config.from_directory(os.path.expanduser(MODEL_DIR))
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    cache_kwargs = {}
    if kv8:
        from exllamav3.cache import CacheLayer_quant
        cache_kwargs = dict(layer_type=CacheLayer_quant, k_bits=8, v_bits=8)
    cache = Cache(model, max_num_tokens=(pool_override or pool), max_history=16, max_batch_size=1,
                  recurrent_dtype=torch.half, **cache_kwargs)
    draft_config = Config.from_directory(os.path.expanduser(drafter_dir))
    draft_model = Model.from_config(draft_config, component="text")
    draft_cache = Cache(draft_model, max_num_tokens=(pool_override or pool), max_history=16, max_batch_size=1,
                        recurrent_dtype=torch.half)
    draft_model.load(progressbar=False, max_batch_size=1, autosplit_no_forward=True,
                     max_chunk_size=int(os.environ.get("MAXCHUNK", "4096")))
    model.load(draft_model=draft_model, progressbar=False, max_batch_size=1,
               autosplit_no_forward=True, max_chunk_size=int(os.environ.get("MAXCHUNK", "4096")))
    if not draft_cache.initialized:
        draft_model._mark_caches_initialized()
    torch.cuda.empty_cache()
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                    draft_model=draft_model, draft_cache=draft_cache,
                    num_draft_tokens=window, dynamic_draft_tokens=dynamic,
                    draft_confidence=conf,
                    hybrid_draft_tokens=hdt,
                    max_chunk_size=int(os.environ.get("MAXCHUNK", "4096")))
    return gen, tokenizer


EFFORT_INSTRUCTIONS = {
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
             "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
             "clarity in the final answer.",
    "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
           "conclusion without unnecessary elaboration.",
    "medium": "",
}


def _tool_json(t):
    """Tool schema as chat_template.jinja's `tool | tojson` renders it:
    sorted keys, default separators, HTML-safe escapes."""
    s = json.dumps(t, sort_keys=True, separators=(", ", ": "))
    return (s.replace("<", "\\u003c").replace(">", "\\u003e")
             .replace("&", "\\u0026").replace("'", "\\u0027"))


def _tools_system_block(tools, effort_instr, sys_content):
    """Reproduce the model's own chat-template tools block verbatim (XML call format)."""
    lines = [_tool_json(t) for t in tools]
    block = (
        "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
        + "\n".join(lines)
        + "\n</tools>"
        + "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
          "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
          "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
          "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
          "- Function calls MUST follow the specified format: an inner <function=...></function> block "
          "must be nested within <tool_call></tool_call> XML tags\n"
          "- Required parameters MUST be specified\n"
          "- You may provide optional reasoning for your function call in natural language BEFORE the "
          "function call, but NOT after\n"
          "- If there is no function call available, answer the question like normal with your current "
          "knowledge and do not tell the user about function calls\n</IMPORTANT>"
    )
    head = effort_instr + "\n\n" if effort_instr else ""
    tail = "\n\n" + sys_content if sys_content else ""
    return f"<|im_start|>system\n{head}{block}{tail}<|im_end|>\n"


TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
THINK_RESCUE_TOKENS = 512  # bounded extra budget for the think-rescue continuation


def unterminated_think(text):
    """True if text ends inside an open <think> block (last open has no close)."""
    return text.rfind(THINK_OPEN) > text.rfind(THINK_CLOSE)


def trim_partial_think_close(text):
    """Drop a trailing partial '</think>' so the injected close stays well-formed."""
    for k in range(len(THINK_CLOSE) - 1, 0, -1):
        if text.endswith(THINK_CLOSE[:k]):
            return text[:-k]
    return text


def parse_tool_block(block):
    """Parse one <tool_call> inner block: Qwen3.8 XML format or Hermes JSON format."""
    s = block.strip()
    if s.startswith("{"):
        try:
            obj = json.loads(s)
            name, args = obj.get("name", ""), obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            if name:
                return {"name": name, "arguments": args}
        except json.JSONDecodeError:
            pass
        return None
    m = re.match(r"<function=(.+?)>", s)
    if m:
        args = {}
        for pm in re.finditer(r"<parameter=([^\s>]+)>\n?(.*?)\n?</parameter>", s, re.S):
            args[pm.group(1)] = pm.group(2)
        return {"name": m.group(1).strip(), "arguments": args}
    return None


def split_tool_calls(text):
    """Non-stream: (clean_content, [calls]) — strips tool_call blocks from content."""
    calls, out, pos = [], [], 0
    while True:
        i = text.find(TOOL_OPEN, pos)
        if i < 0:
            break
        j = text.find(TOOL_CLOSE, i)
        end = j if j >= 0 else len(text)
        out.append(text[pos:i])
        c = parse_tool_block(text[i + len(TOOL_OPEN):end])
        if c:
            calls.append(c)
        pos = end + len(TOOL_CLOSE) if j >= 0 else len(text)
    out.append(text[pos:])
    return "".join(out).strip("\n"), calls


class ToolStreamFilter:
    """Streaming router: tool_call blocks -> tool_calls deltas, everything else -> content."""

    def __init__(self, emit_content, emit_tool):
        self.buf, self.inside = "", False
        self._ec, self._et = emit_content, emit_tool
        self.has_tools = False

    def feed(self, delta):
        self.buf += delta
        while True:
            if not self.inside:
                i = self.buf.find(TOOL_OPEN)
                if i >= 0:
                    if self.buf[:i]:
                        self._ec(self.buf[:i])
                    self.buf = self.buf[i + len(TOOL_OPEN):]
                    self.inside = True
                    continue
                keep = max(0, len(self.buf) - (len(TOOL_OPEN) - 1))  # hold partial open tag
                if keep:
                    self._ec(self.buf[:keep])
                    self.buf = self.buf[keep:]
                break
            j = self.buf.find(TOOL_CLOSE)
            if j >= 0:
                c = parse_tool_block(self.buf[:j])
                if c:
                    self.has_tools = True
                    self._et(c)
                self.buf = self.buf[j + len(TOOL_CLOSE):]
                self.inside = False
                continue
            break  # hold everything while inside a block

    def flush(self):
        if self.inside:  # unterminated block at EOS: best-effort parse
            c = parse_tool_block(self.buf)
            if c:
                self.has_tools = True
                self._et(c)
        elif self.buf:
            self._ec(self.buf)
        self.buf = ""


def _tool_call_xml(fn):
    """One tool_call dict -> Qwen3.8 XML block (mirrors chat_template.jinja)."""
    name = fn.get("name", "")
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"raw": args}
    if not isinstance(args, dict):
        args = {"raw": json.dumps(args, ensure_ascii=False)}
    inner = ""
    for k, v in args.items():
        val = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        inner += f"<parameter={k}>\n{val}\n</parameter>\n"
    return f"<tool_call>\n<function={name}>\n{inner}</function>\n</tool_call>"


def _assistant_body(m):
    """Assistant message -> chat_template.jinja default rendering:
    '<think>\\n{reasoning}\\n</think>\\n\\n{content}' (think scaffold always
    present, reasoning from reasoning_content or extracted from a raw think
    block embedded in content) + tool_call XML blocks."""
    raw = m.get("content") or ""
    raw = raw.strip() if isinstance(raw, str) else str(raw)
    reasoning = m.get("reasoning_content") or ""
    reasoning = reasoning.strip() if isinstance(reasoning, str) else str(reasoning)
    i, j = raw.find(THINK_OPEN), raw.find(THINK_CLOSE)
    if i >= 0 and j > i:  # normalize our own raw-think replay into the scaffold
        if not reasoning:
            reasoning = raw[i + len(THINK_OPEN):j].strip()
        raw = (raw[:i] + raw[j + len(THINK_CLOSE):]).strip()
    body = f"<think>\n{reasoning}\n</think>\n\n{raw}"
    calls = m.get("tool_calls") or []
    for k, tc in enumerate(calls):
        fn = tc.get("function") or tc
        sep = "\n\n" if (k == 0 and raw) else ("\n" if k else "")
        body += sep + _tool_call_xml(fn)
    return body


def chat_prompt(messages, embeddings=None, enable_thinking=True, reasoning_effort="xhigh",
                tools=None):
    """Qwen chat template; image aliases spliced before text; effort + thinking per template."""
    aliases = [e.text_alias for e in (embeddings or [])]
    # template resolves reasoning effort only when thinking is enabled
    eff = EFFORT_INSTRUCTIONS.get(reasoning_effort, "") if enable_thinking else ""
    parts = []
    sysmsg = [m["content"] for m in messages if m.get("role") == "system"]
    if tools:
        parts.append(_tools_system_block(tools, eff, sysmsg[0] if sysmsg else ""))
    elif sysmsg:
        sys = (eff + "\n\n" if eff else "") + sysmsg[0]
        parts.append(f"<|im_start|>system\n{sys}<|im_end|>\n")
    elif eff:
        parts.append(f"<|im_start|>system\n{eff}<|im_end|>\n")
    convo = [m for m in messages if m.get("role") != "system"]
    first_user = True
    for idx, m in enumerate(convo):
        role = m.get("role")
        if role == "tool":
            # consecutive tool results merge into ONE user turn of <tool_response> blocks
            prev_tool = idx > 0 and convo[idx - 1].get("role") == "tool"
            next_tool = idx + 1 < len(convo) and convo[idx + 1].get("role") == "tool"
            if not prev_tool:
                parts.append("<|im_start|>user")
            body = m.get("content") or ""
            body = body.strip() if isinstance(body, str) else str(body)
            parts.append(f"\n<tool_response>\n{body}\n</tool_response>")
            if not next_tool:
                parts.append("<|im_end|>\n")
            continue
        role = "user" if role != "assistant" else "assistant"
        if role == "assistant":
            body = _assistant_body(m)
        else:
            body = m.get("content") or ""
            body = body.strip() if isinstance(body, str) else str(body)
        if role == "user" and first_user and aliases:
            body = "\n".join(aliases) + "\n" + body
            first_user = False
        parts.append(f"<|im_start|>{role}\n{body}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    # chat_template.jinja add_generation_prompt: thinking on -> prefill the open
    # <think> block (the model continues it; it cannot decline to think); off ->
    # the closed empty block makes it answer directly.
    if enable_thinking:
        parts.append("<think>\n")
    else:
        parts.append("<think>\n\n</think>\n\n")  # exact template off-switch
    return "".join(parts)


class Server:
    def __init__(self, lane, dynamic, conf, window=None, kv8=False, pool_override=None, hdt=0):
        self.lane = lane
        print(f"[serve] loading target {MODEL_DIR}", flush=True)
        t0 = time.time()
        self.gen, self.tokenizer = build_generator(lane, dynamic, conf, window, kv8=kv8, pool_override=pool_override, hdt=hdt)
        self.pool_tokens = pool_override or LANES[lane][2]
        self.default_window = self.gen.num_draft_tokens
        self.lock = threading.Lock()
        self.eos = [self.tokenizer.eos_token_id, self.tokenizer.single_id("<|im_end|>")]
        self.vision_model = None
        self.vision_lock = threading.Lock()
        print(f"[serve] lane={lane} window={self.gen.num_draft_tokens} (default {self.default_window}) dynamic={dynamic} conf={conf} "
              f"loaded in {time.time()-t0:.0f}s", flush=True)
        # warmup (think_rescue off: keep boot fast, answer completeness irrelevant)
        # long enough to hit every dynamic verify/draft shape once (graph capture)
        self.generate([{"role": "user", "content": "Say OK."}], 8, think_rescue=False)
        t0 = time.time()
        self.generate(
            [{"role": "user", "content":
              "Write a detailed step by step plan for building a small greenhouse. "
              "Cover materials, tools, budget, timeline, ventilation, watering, and common mistakes. "
              "Vary sentence lengths naturally as you go, mixing short confirmations with longer "
              "explanations, so the drafting engine sees many different token patterns."}],
            256, think_rescue=False)
        print(f"[serve] warmup done (shape walk {time.time()-t0:.1f}s)", flush=True)

    # ---------- vision ----------

    def _get_vision(self):
        with self.vision_lock:
            if self.vision_model is None:
                print("[serve] loading vision tower (on-demand, stays resident)", flush=True)
                t0 = time.time()
                vm = Model.from_config(self.gen.model.config, component="vision")
                vm.load()
                self.vision_model = vm
                print(f"[serve] vision tower loaded in {time.time()-t0:.0f}s", flush=True)
            return self.vision_model

    def embed_image(self, pil_image):
        vm = self._get_vision()
        with self.lock:
            ie = vm.get_image_embeddings(tokenizer=self.tokenizer, image=pil_image)
        # 128K fp16 KV pool leaves no room to keep the tower resident:
        # unload after embed (3s reload per image request)
        with self.vision_lock:
            if self.vision_model is not None:
                self.vision_model.unload()
                self.vision_model = None
                torch.cuda.empty_cache()
        return ie

    @staticmethod
    def extract_images(messages):
        """Return (text_messages, images) — OpenAI multimodal content arrays supported."""
        out_msgs, images = [], []
        for m in messages:
            if isinstance(m.get("content"), list):
                text_parts = []
                for part in m["content"]:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url" and VISION_ENABLED:
                        url = part.get("image_url", {}).get("url", "")
                        images.append(url)
                m2 = dict(m)
                m2["content"] = "\n".join(text_parts)
                out_msgs.append(m2)
            else:
                out_msgs.append(m)
        return out_msgs, images

    @staticmethod
    def load_image(url):
        if url.startswith("data:"):
            header, b64 = url.split(",", 1)
            raw = base64.b64decode(b64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        if url.startswith("http://") or url.startswith("https://"):
            with urllib.request.urlopen(url, timeout=30) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        return Image.open(url).convert("RGB")

    def prepare_images(self, messages):
        """Decode + embed ALL images up front (Bug 8). A corrupt/truncated image
        raises ImageDataError(messages_index) BEFORE any response bytes go out,
        so the client gets a clean 400 instead of a dead connection it can only
        blind-retry with the same bad bytes (the ECONNRESET storm loop)."""
        pairs = []
        for i, m in enumerate(messages):
            if isinstance(m.get("content"), list):
                for part in m["content"]:
                    if part.get("type") == "image_url" and VISION_ENABLED:
                        pairs.append((i, part.get("image_url", {}).get("url", "")))
        embeddings = []
        for i, url in pairs:
            try:
                embeddings.append(self.embed_image(self.load_image(url)))
            except Exception as e:
                raise ImageDataError(i, e)
        return embeddings or None

    # ---------- generation ----------

    def _gen_pass(self, input_ids, max_new_tokens, sampler, embeddings, emit=None,
                  min_new_tokens=0):
        """Run one Job to completion; returns (text, decode_s, n_tokens, hit_cap, eos_reason).
        t0-late fallback keeps t/s defined even for zero/one-token passes."""
        job = Job(input_ids=input_ids, max_new_tokens=max_new_tokens,
                  min_new_tokens=min_new_tokens,
                  decode_special_tokens=True, sampler=sampler,
                  stop_conditions=self.eos, embeddings=embeddings)
        with self.lock:
            self.gen.enqueue(job)
            out, t0, t_start, eos_reason = "", None, time.time(), None
            while self.gen.num_remaining_jobs():
                results = self.gen.iterate()
                if t0 is None and out:
                    t0 = time.time()
                for r in results:
                    if r.get("eos_reason"):
                        eos_reason = r["eos_reason"]
                    d = r.get("text", "")
                    if d:
                        if emit:
                            emit(d)
                        out += d
            dt = max(1e-6, time.time() - (t0 if t0 is not None else t_start))
        n = len(self.tokenizer.encode(out, encode_special_tokens=True).flatten())
        return out, dt, n, n >= max_new_tokens, eos_reason

    def generate(self, messages, max_tokens, temperature=0.0, emit=None,
                 enable_thinking=True, reasoning_effort="xhigh", tools=None,
                 think_rescue=True, top_p=1.0, top_k=0, min_p=0.0, min_tokens=0,
                 embeddings=None):
        """Full generation (streaming via emit(text_delta) if given).
        Returns (text, tps, total_completion_tokens, hit_cap_of_last_pass).

        Sampling: temperature/top_p/top_k/min_p reach the engine via ComboSampler
        (a bare temperature= kwarg on Job is silently swallowed by **kwargs).

        min_tokens: engine-native min_new_tokens — stop tokens are suppressed
        until that many tokens have been sampled.

        Think rescue: if thinking was on and generation ended inside an open
        <think> block (token cap hit mid-think, or premature EOS), the client
        strips think text and would see an EMPTY answer. Close the block and
        run a bounded continuation pass that produces the actual answer;
        deltas keep streaming and completion_tokens are summed across passes.
        Opt out per request with {"think_rescue": false}."""
        messages, _ = self.extract_images(messages)
        if embeddings is None:
            embeddings = self.prepare_images(messages)  # may raise ImageDataError
        prompt = chat_prompt(messages, embeddings, enable_thinking, reasoning_effort,
                             tools)
        input_ids = self.tokenizer.encode(prompt, encode_special_tokens=True,
                                          embeddings=embeddings)
        # Edge guard (crash 2026-11-13): at the physical pool cap the drafter's
        # eager window-shift round walks past its ring (Xid 13, fault 0xc81eb60,
        # deterministic at ~160.7k/163840). Never let a sequence reach that
        # zone: keep a hard margin below the pool, reserve the rescue budget
        # up front, and clamp max_tokens to what provably fits.
        EDGE_MARGIN = 4096
        budget = self.pool_tokens - EDGE_MARGIN
        n_tok = input_ids.shape[-1]
        reserve = (THINK_RESCUE_TOKENS + 128) if (think_rescue and enable_thinking) else 128
        cap = budget - n_tok - reserve
        if cap < 1:
            raise AssertionError(
                f"prompt {n_tok} tokens exceeds usable context {budget} "
                f"(pool {self.pool_tokens}, edge margin {EDGE_MARGIN})")
        if max_tokens > cap:
            print(f"[serve] edge clamp: max_tokens {max_tokens} -> {cap} "
                  f"(prompt {n_tok}, pool {self.pool_tokens})", flush=True)
            max_tokens = cap
        sampler = ComboSampler(temperature=temperature, top_p=top_p, top_k=top_k,
                               min_p=min_p)
        min_new = max(0, min(int(min_tokens), max_tokens))
        out = "<think>\n" if enable_thinking else ""  # opener is prefilled in the
        if out and emit:                              # prompt; re-emit it so the
            emit(out)                                 # client sees complete tags
        gen1, dt, n, capped, eos_reason = self._gen_pass(
            input_ids, max_tokens, sampler, embeddings, emit, min_new_tokens=min_new)
        out += gen1
        if enable_thinking and n == 0:
            # instant-EOS rescue (Bug 5): the model stopped on the very first
            # token despite the prefilled think opener. Suppress stop tokens for
            # a bounded retry so at least a well-formed think block comes back.
            print(f"[serve] empty completion rescue: 0 tokens, eos_reason="
                  f"{eos_reason}, {len(messages)} msgs, thinking on", flush=True)
            out2, dt2, n2, capped, _ = self._gen_pass(
                input_ids, max_tokens, sampler, embeddings, emit,
                min_new_tokens=max(32, min_new))
            out, n, dt = out + out2, n + n2, dt + dt2
        if think_rescue and enable_thinking and unterminated_think(out):
            bridge = "</think>\n\n"
            cont_prompt = trim_partial_think_close(out) + bridge
            input_ids2 = self.tokenizer.encode(prompt + cont_prompt,
                                               encode_special_tokens=True,
                                               embeddings=embeddings)
            if emit:
                emit(bridge)
            # re-encoding the streamed text can drift a few tokens past the
            # reserve computed above; re-check against the edge budget here
            rescue_cap = min(THINK_RESCUE_TOKENS, budget - input_ids2.shape[-1] - 64)
            if rescue_cap < 16:
                print(f"[serve] rescue skipped: no edge budget "
                      f"(prompt2 {input_ids2.shape[-1]}, budget {budget})", flush=True)
            else:
                out2, dt2, n2, capped, _ = self._gen_pass(
                    input_ids2, rescue_cap, sampler, embeddings, emit)
                out = cont_prompt + out2  # trimmed think + injected close + answer
                n = n + n2                # completion tokens summed across passes
                dt += dt2
                if rescue_cap < THINK_RESCUE_TOKENS:
                    capped = True
                del rescue_cap
        tps = n / dt
        return out, tps, n, capped


SRV = None


class ImageDataError(Exception):
    """An image attached to messages[index] failed to load/decode (Bug 8)."""
    def __init__(self, index, cause):
        super().__init__(f"invalid image data at messages[{index}]: {cause}")
        self.index = index


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("[http] " + fmt % args, flush=True)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            lane = SRV.lane if SRV else "dspine"
            self._json(200, {"object": "list", "data": [{
                "id": "qwen3.8-27b-exl3-4bpw-prose", "object": "model",
                "owned_by": "catonooka",
                "context_length": SRV.gen.cache.max_num_tokens,
                "max_completion_tokens": 32768,
                "capabilities": {"vision": VISION_ENABLED, "tools": True,
                                 "thinking": ["off", "low", "medium", "xhigh"],
                                 "streaming": True}}]})
        elif self.path == "/health":
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "bad json"})
            return
        messages = req.get("messages") or []
        if not messages:
            self._json(400, {"error": "messages required"})
            return
        # BUG1 fix: honor max_completion_tokens (OpenAI current) alongside legacy
        # max_tokens; whichever is present wins, max_completion_tokens on ties.
        mct, mt = req.get("max_completion_tokens"), req.get("max_tokens")
        max_tokens = int(mct if mct is not None else (mt if mt is not None else 1024))
        max_tokens = max(1, min(max_tokens, 32768))
        temperature = float(req.get("temperature") or 0.0)
        temperature = 0.0 if temperature < 1e-4 else temperature
        top_p = min(1.0, max(0.0, float(req.get("top_p") or 1.0)))
        top_k = max(0, int(req.get("top_k") or 0))
        min_p = min(1.0, max(0.0, float(req.get("min_p") or 0.0)))
        # min_tokens: OpenAI-style minimum completion length; engine suppresses
        # stop tokens until it is reached (min_completion_tokens also accepted)
        min_tokens = int(req.get("min_tokens") or req.get("min_completion_tokens") or 0)
        stream = bool(req.get("stream"))
        tools = req.get("tools") or None
        # think-rescue continuation (close unterminated <think>, answer with a
        # bounded +512-token pass); default on, opt out with {"think_rescue": false}
        think_rescue = bool(req.get("think_rescue", True))
        # thinking controls: OpenAI-style reasoning_effort/enable_thinking +
        # z.ai-style thinking.type parity; "none" = alias for thinking off
        reasoning_effort = req.get("reasoning_effort") or "xhigh"
        if reasoning_effort == "none":
            reasoning_effort, enable_thinking = "medium", False
        else:
            enable_thinking = req.get("enable_thinking", True)
        if reasoning_effort not in ("xhigh", "medium", "low"):
            self._json(400, {"error": f"Unexpected reasoning effort {reasoning_effort}. "
                                      "Supported types are xhigh (default), medium, and low."})
            return
        th = req.get("thinking")
        if isinstance(th, dict) and th.get("type") == "disabled":
            enable_thinking = False
        model_id = "qwen3.8-27b-exl3-4bpw-prose"
        t0 = time.time()

        # Bug 8: decode/validate ALL images BEFORE any bytes go out — a corrupt
        # image becomes a clean 400 instead of a mid-stream dead connection.
        try:
            embeddings = SRV.prepare_images(messages)
        except ImageDataError as e:
            self._json(400, {"error": f"invalid image data at messages[{e.index}]"})
            return
        except Exception as e:
            self._json(500, {"error": f"image processing failed: {e}"})
            return

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def sse(obj):
                data = b"data: " + json.dumps(obj).encode() + b"\n\n"
                self.wfile.write(("%x\r\n" % len(data)).encode() + data + b"\r\n")
                self.wfile.flush()

            def chunk(delta_text, finish=None):
                sse({"id": "cmpl-prose", "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": {"content": delta_text},
                                  "finish_reason": finish}]})

            def chunk_tool(idx, call):
                sse({"id": "cmpl-prose", "object": "chat.completion.chunk",
                     "choices": [{"index": 0,
                                  "delta": {"tool_calls": [{
                                      "index": idx, "type": "function",
                                      "id": "call_" + uuid.uuid4().hex[:24],
                                      "function": {"name": call["name"],
                                                   "arguments": json.dumps(call["arguments"])}}]},
                                  "finish_reason": None}]})

            def finish_reason(capped, has_tools):
                if has_tools:
                    return "tool_calls"
                return "length" if capped else "stop"

            try:
                chunk("")
                tool_idx = [0]
                flt = ToolStreamFilter(
                    emit_content=lambda d: chunk(d),
                    emit_tool=lambda c: (chunk_tool(tool_idx[0], c),
                                         tool_idx.__setitem__(0, tool_idx[0] + 1)))
                text, tps, n, capped = SRV.generate(
                    messages, max_tokens, temperature,
                    emit=lambda d: flt.feed(d),
                    enable_thinking=enable_thinking, reasoning_effort=reasoning_effort,
                    tools=tools, think_rescue=think_rescue, top_p=top_p, top_k=top_k,
                    min_p=min_p, min_tokens=min_tokens, embeddings=embeddings)
                flt.flush()
                chunk("", finish=finish_reason(capped, flt.has_tools))
                sse({"stats": {"tokens": n, "tps": round(tps, 1),
                               "wall_s": round(time.time() - t0, 2)}})
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except torch.cuda.OutOfMemoryError:
                sse({"error": "CUDA OOM: prompt + max_tokens exceed the context pool"})
                self.wfile.write(b"0\r\n\r\n")
            except Exception as e:
                # Bug 8 principle: a lane-internal error must surface as data,
                # never as a dead connection the client can only blind-retry.
                print(f"[serve] stream error: {type(e).__name__}: {e}", flush=True)
                import traceback as _tb; _tb.print_exc(file=__import__("sys").stdout); print(flush=True)
                try:
                    sse({"error": f"{type(e).__name__}: {e}"})
                    self.wfile.write(b"0\r\n\r\n")
                except Exception:
                    pass
        else:
            try:
                text, tps, n, capped = SRV.generate(messages, max_tokens, temperature,
                                                    enable_thinking=enable_thinking,
                                                    reasoning_effort=reasoning_effort,
                                                    tools=tools,
                                                    think_rescue=think_rescue,
                                                    top_p=top_p, top_k=top_k,
                                                    min_p=min_p,
                                                    min_tokens=min_tokens,
                                                    embeddings=embeddings)
            except torch.cuda.OutOfMemoryError:
                self._json(503, {"error": "CUDA OOM: prompt + max_tokens exceed the context pool"})
                return
            except AssertionError as e:
                self._json(400, {"error": f"context pool exceeded: {e}"})
                return
            except Exception as e:
                # Bug 8 principle: JSON error, never a killed connection
                print(f"[serve] request error: {type(e).__name__}: {e}", flush=True)
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
                return
            content, calls = split_tool_calls(text)
            if calls:
                finish = "tool_calls"
                tool_calls = [{"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                               "function": {"name": c["name"],
                                            "arguments": json.dumps(c["arguments"])}}
                              for c in calls]
            else:
                finish = "length" if capped else "stop"
                tool_calls = None
            self._json(200, {
                "id": "cmpl-prose", "object": "chat.completion",
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": content,
                                         "tool_calls": tool_calls},
                             "finish_reason": finish}],
                "usage": {"completion_tokens": n},
                "stats": {"tokens": n, "tps": round(tps, 1),
                          "wall_s": round(time.time() - t0, 2)}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--lane", default="dspine",
                    choices=list(LANES) + ["all"])
    ap.add_argument("--no-dynamic", action="store_true")
    ap.add_argument("--kv8", action="store_true", help="8-bit K/V quantized target KV cache")
    ap.add_argument("--pool", type=int, default=None, help="override context pool tokens (multiple of PAGE_SIZE)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--window", type=int, default=None,
                    help="drafter window override (bw_reps: prose w8, code w15)")
    ap.add_argument("--hdt", type=int, default=0,
                    help="hybrid_draft_tokens: fire a long suffix-match draft when context repeats (0 = off)")
    args = ap.parse_args()
    global SRV
    SRV = Server(args.lane, not args.no_dynamic, args.conf, args.window, kv8=args.kv8, pool_override=args.pool, hdt=args.hdt)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] listening on http://{args.host}:{args.port}/v1/chat/completions", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
