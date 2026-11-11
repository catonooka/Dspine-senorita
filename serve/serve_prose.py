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
import argparse, json, os, sys, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("DSPINE_GRAPH", "1")
sys.path.insert(0, os.path.expanduser("~/exl3_pipeline/exllamav3"))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job

VISION_ENABLED = True
try:
    from PIL import Image
    import base64, io, urllib.request
except ImportError:
    VISION_ENABLED = False

LANES = {
    "dflash2":     ("~/exl3_pipeline/output_exl3/Drafter-DFlash2-EXL3-5.0bpw-h8", 7),
    "dflash2-bf16":("~/exl3_pipeline/raw_dflash2", 7),
    "dspine":      ("~/exl3_pipeline/raw_drafter", 15),
    "dspine-exl3": ("~/exl3_pipeline/output_exl3/Drafter-DSpine-EXL3-5.0bpw-h8", 15),
}
MODEL_DIR = "~/exl3_pipeline/output_exl3/Model-27B-EXL3-4.0bpw-h8"


def build_generator(lane, dynamic, conf):
    drafter_dir, window = LANES[lane]
    config = Config.from_directory(os.path.expanduser(MODEL_DIR))
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=122880, max_history=16, max_batch_size=1,
                  recurrent_dtype=torch.half)
    draft_config = Config.from_directory(os.path.expanduser(drafter_dir))
    draft_model = Model.from_config(draft_config, component="text")
    draft_cache = Cache(draft_model, max_num_tokens=122880, max_history=16, max_batch_size=1,
                        recurrent_dtype=torch.half)
    draft_model.load(progressbar=False, max_batch_size=1, autosplit_no_forward=True,
                     max_chunk_size=2048)
    model.load(draft_model=draft_model, progressbar=False, max_batch_size=1,
               autosplit_no_forward=True, max_chunk_size=2048)
    if not draft_cache.initialized:
        draft_model._mark_caches_initialized()
    torch.cuda.empty_cache()
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                    draft_model=draft_model, draft_cache=draft_cache,
                    num_draft_tokens=window, dynamic_draft_tokens=dynamic,
                    draft_confidence=conf)
    return gen, tokenizer


EFFORT_INSTRUCTIONS = {
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
             "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
             "clarity in the final answer.",
    "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
           "conclusion without unnecessary elaboration.",
    "medium": "",
}


def chat_prompt(messages, embeddings=None, enable_thinking=True, reasoning_effort="xhigh"):
    """Qwen chat template; image aliases spliced before text; effort + thinking per template."""
    aliases = [e.text_alias for e in (embeddings or [])]
    eff = EFFORT_INSTRUCTIONS.get(reasoning_effort, "")
    parts = []
    sysmsg = [m["content"] for m in messages if m.get("role") == "system"]
    if sysmsg:
        sys = sysmsg[0] + ("\n\n" + eff if eff else "")
        parts.append(f"<|im_start|>system\n{sys}<|im_end|>\n")
    elif eff:
        parts.append(f"<|im_start|>system\n{eff}<|im_end|>\n")
    first_user = True
    for m in messages:
        if m.get("role") == "system":
            continue
        role = "user" if m.get("role") != "assistant" else "assistant"
        body = m["content"]
        if role == "user" and first_user and aliases:
            body = "\n".join(aliases) + "\n" + body
            first_user = False
        parts.append(f"<|im_start|>{role}\n{body}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    if not enable_thinking:
        parts.append("<think>\n\n</think>\n\n")  # exact template off-switch
    return "".join(parts)


class Server:
    def __init__(self, lane, dynamic, conf):
        print(f"[serve] loading target {MODEL_DIR}", flush=True)
        t0 = time.time()
        self.gen, self.tokenizer = build_generator(lane, dynamic, conf)
        self.lock = threading.Lock()
        self.eos = [self.tokenizer.eos_token_id, self.tokenizer.single_id("<|im_end|>")]
        self.vision_model = None
        self.vision_lock = threading.Lock()
        print(f"[serve] lane={lane} window={LANES[lane][1]} dynamic={dynamic} conf={conf} "
              f"loaded in {time.time()-t0:.0f}s", flush=True)
        # warmup
        self.generate([{"role": "user", "content": "Say OK."}], 8)
        print("[serve] warmup done", flush=True)

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

    # ---------- generation ----------

    def generate(self, messages, max_tokens, temperature=0.0,
                 enable_thinking=True, reasoning_effort="xhigh"):
        """Full generation, returns (text, tps, n_tokens)."""
        messages, img_urls = self.extract_images(messages)
        embeddings = [self.embed_image(self.load_image(u)) for u in img_urls]
        prompt = chat_prompt(messages, embeddings, enable_thinking, reasoning_effort)
        input_ids = self.tokenizer.encode(prompt, encode_special_tokens=True,
                                          embeddings=embeddings or None)
        job = Job(input_ids=input_ids, max_new_tokens=max_tokens,
                  decode_special_tokens=True, temperature=temperature,
                  stop_conditions=self.eos, embeddings=embeddings or None)
        with self.lock:
            self.gen.enqueue(job)
            out, t0, t_start = "", None, time.time()
            while self.gen.num_remaining_jobs():
                results = self.gen.iterate()
                if t0 is None and out:
                    t0 = time.time()
                for r in results:
                    out += r.get("text", "")
            dt = max(1e-6, time.time() - (t0 if t0 is not None else t_start))
        n = len(self.tokenizer.encode(out, encode_special_tokens=True).flatten())
        return out, n / dt, n

    def generate_stream(self, messages, max_tokens, temperature=0.0, emit=None,
                        enable_thinking=True, reasoning_effort="xhigh"):
        """Streaming generation; emit(text_delta) called per chunk. Returns stats."""
        messages, img_urls = self.extract_images(messages)
        embeddings = [self.embed_image(self.load_image(u)) for u in img_urls]
        prompt = chat_prompt(messages, embeddings, enable_thinking, reasoning_effort)
        input_ids = self.tokenizer.encode(prompt, encode_special_tokens=True,
                                          embeddings=embeddings or None)
        job = Job(input_ids=input_ids, max_new_tokens=max_tokens,
                  decode_special_tokens=True, temperature=temperature,
                  stop_conditions=self.eos, embeddings=embeddings or None)
        with self.lock:
            self.gen.enqueue(job)
            out, t0, t_start = "", None, time.time()
            while self.gen.num_remaining_jobs():
                results = self.gen.iterate()
                if t0 is None and out:
                    t0 = time.time()
                for r in results:
                    d = r.get("text", "")
                    if d and emit:
                        emit(d)
                    out += d
            dt = max(1e-6, time.time() - (t0 if t0 is not None else t_start))
        n = len(self.tokenizer.encode(out, encode_special_tokens=True).flatten())
        return out, n / dt, n


SRV = None


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
            self._json(200, {"object": "list", "data": [
                {"id": "qwen3.8-27b-exl3-4bpw-prose", "object": "model"}]})
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
        max_tokens = int(req.get("max_tokens") or 1024)
        temperature = float(req.get("temperature") or 0.0)
        temperature = 0.0 if temperature < 1e-4 else temperature
        stream = bool(req.get("stream"))
        # thinking controls: OpenAI-style reasoning_effort/enable_thinking +
        # z.ai-style thinking.type parity
        reasoning_effort = req.get("reasoning_effort") or "xhigh"
        if reasoning_effort not in ("xhigh", "medium", "low"):
            self._json(400, {"error": f"Unexpected reasoning effort {reasoning_effort}. "
                                      "Supported types are xhigh (default), medium, and low."})
            return
        enable_thinking = req.get("enable_thinking", True)
        th = req.get("thinking")
        if isinstance(th, dict) and th.get("type") == "disabled":
            enable_thinking = False
        model_id = "qwen3.8-27b-exl3-4bpw-prose"
        t0 = time.time()

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
            try:
                chunk("")
                text, tps, n = SRV.generate_stream(
                    messages, max_tokens, temperature,
                    emit=lambda d: chunk(d),
                    enable_thinking=enable_thinking, reasoning_effort=reasoning_effort)
                chunk("", finish="stop")
                sse({"stats": {"tokens": n, "tps": round(tps, 1),
                               "wall_s": round(time.time() - t0, 2)}})
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except torch.cuda.OutOfMemoryError:
                sse({"error": "CUDA OOM: prompt + max_tokens exceed the context pool"})
                self.wfile.write(b"0\r\n\r\n")
        else:
            try:
                text, tps, n = SRV.generate(messages, max_tokens, temperature,
                                            enable_thinking=enable_thinking,
                                            reasoning_effort=reasoning_effort)
            except torch.cuda.OutOfMemoryError:
                self._json(503, {"error": "CUDA OOM: prompt + max_tokens exceed the context pool"})
                return
            except AssertionError as e:
                self._json(400, {"error": f"context pool exceeded: {e}"})
                return
            self._json(200, {
                "id": "cmpl-prose", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
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
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    global SRV
    SRV = Server(args.lane, not args.no_dynamic, args.conf)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] listening on http://{args.host}:{args.port}/v1/chat/completions", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
