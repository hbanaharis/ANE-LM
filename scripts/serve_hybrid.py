#!/usr/bin/env python3
"""Hybrid ANE/CPU inference server for Qwen3.5-4B.

Serves the model via HTTP with streaming text generation.
Pipeline: tokenize → embed(CPU) → [ANE chunk → CPU attn] x8 → ANE final → LM head(ANE) → sample → detokenize

Usage:
    python scripts/serve_hybrid.py [--port 8000] [--host 0.0.0.0]
"""

import argparse
import json
import math
import os
import sys
import time
import ctypes
import ctypes.util
from typing import Optional

import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

# ============ Model Config ============

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.5-4B")
HYBRID_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.5-4B-hybrid")
LM_HEAD_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.5-4B-coreml", "lm_head_chunk_0.mlpackage")

HIDDEN = 2560
NUM_HEADS = 16
NUM_KV_HEADS = 4
HEAD_DIM = 256
GROUPS = NUM_HEADS // NUM_KV_HEADS
ROPE_DIM = 64
HALF_ROT = ROPE_DIM // 2
ATTN_DIM = NUM_HEADS * HEAD_DIM
CONTEXT = 4096
N_DN_PER_BLOCK = 3
LIN_QKV_DIM = 8192
LIN_NUM_VAL_HEADS = 32
LIN_KEY_DIM = 128
LIN_VAL_DIM = 128
CONV_KERNEL = 4
VOCAB_SIZE = 248320

# RoPE
ROPE_THETA = 10000000.0
_inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, ROPE_DIM, 2, dtype=np.float32) / ROPE_DIM))
_positions = np.arange(CONTEXT, dtype=np.float32)
_freqs = np.outer(_positions, _inv_freq)
ROPE_COS = np.cos(_freqs).astype(np.float32)
ROPE_SIN = np.sin(_freqs).astype(np.float32)

# BLAS
_accel = ctypes.CDLL(ctypes.util.find_library("Accelerate"))
_cblas_sgemm = _accel.cblas_sgemm
CblasRowMajor = 101
CblasNoTrans = 111
CblasTrans = 112


def sgemm(A, B, C, M, N, K, transB=False):
    lda = K
    if transB:
        ldb = K
        _cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                     M, N, K, ctypes.c_float(1.0),
                     A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), lda,
                     B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ldb,
                     ctypes.c_float(0.0),
                     C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N)
    else:
        ldb = N
        _cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     M, N, K, ctypes.c_float(1.0),
                     A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), lda,
                     B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ldb,
                     ctypes.c_float(0.0),
                     C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N)


def cpu_attention(q_gate_fp16, k_fp16, v_fp16, pos, kv_caches, layer_idx):
    k_cache, v_cache = kv_caches[layer_idx]
    q_gate = q_gate_fp16.astype(np.float32)
    k = k_fp16.astype(np.float32)
    v = v_fp16.astype(np.float32)

    q = q_gate[:, :HEAD_DIM]
    gate = q_gate[:, HEAD_DIM:]

    cos = ROPE_COS[pos]
    sin = ROPE_SIN[pos]
    q1 = q[:, :HALF_ROT].copy()
    q2 = q[:, HALF_ROT:ROPE_DIM].copy()
    q[:, :HALF_ROT] = q1 * cos - q2 * sin
    q[:, HALF_ROT:ROPE_DIM] = q2 * cos + q1 * sin
    k1 = k[:, :HALF_ROT].copy()
    k2 = k[:, HALF_ROT:ROPE_DIM].copy()
    k[:, :HALF_ROT] = k1 * cos - k2 * sin
    k[:, HALF_ROT:ROPE_DIM] = k2 * cos + k1 * sin

    k_cache[:, pos, :] = k
    v_cache[:, pos, :] = v

    seq_len = pos + 1
    scale = 1.0 / math.sqrt(HEAD_DIM)
    attn_out = np.empty((NUM_HEADS, HEAD_DIM), dtype=np.float32)

    for g in range(NUM_KV_HEADS):
        q_start = g * GROUPS
        q_block = np.ascontiguousarray(q[q_start:q_start + GROUPS])
        k_block = np.ascontiguousarray(k_cache[g, :seq_len, :])
        v_block = np.ascontiguousarray(v_cache[g, :seq_len, :])

        scores = np.empty((GROUPS, seq_len), dtype=np.float32)
        sgemm(q_block, k_block, scores, GROUPS, seq_len, HEAD_DIM, transB=True)
        scores *= scale

        scores -= scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        out_block = np.empty((GROUPS, HEAD_DIM), dtype=np.float32)
        sgemm(attn_weights, v_block, out_block, GROUPS, HEAD_DIM, seq_len)
        attn_out[q_start:q_start + GROUPS] = out_block

    sigmoid_gate = 1.0 / (1.0 + np.exp(-gate))
    attn_out *= sigmoid_gate
    return attn_out.astype(np.float16).reshape(1, ATTN_DIM)


# ============ Model Wrapper ============

class HybridModel:
    def __init__(self):
        self.tokenizer = None
        self.embed_weights = None
        self.chunk_0 = None
        self.mid_chunks = []
        self.final_chunk = None
        self.lm_head = None
        self.stop_token_ids = set()

    def load(self):
        t0 = time.time()
        compute_unit = ct.ComputeUnit.CPU_AND_NE

        # Tokenizer
        print("[Hybrid] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        self.stop_token_ids = set()
        eos = self.tokenizer.eos_token_id
        if isinstance(eos, list):
            self.stop_token_ids.update(eos)
        elif eos is not None:
            self.stop_token_ids.add(eos)
        for tok_str in ("<|endoftext|>", "<|im_end|>", "<|end|>"):
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok_str)
                if tid is not None and tid != self.tokenizer.unk_token_id:
                    self.stop_token_ids.add(tid)
            except Exception:
                pass

        # Embedding weights (CPU lookup)
        print("[Hybrid] Loading embedding weights...")
        from safetensors import safe_open
        index_path = os.path.join(MODEL_DIR, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        # Try both possible prefixes
        embed_key = None
        for pfx in ["model.language_model.", "model.", ""]:
            k = f"{pfx}embed_tokens.weight"
            if k in weight_map:
                embed_key = k
                break
        if embed_key is None:
            raise ValueError("Cannot find embed_tokens.weight")

        shard_file = weight_map[embed_key]
        shard_path = os.path.join(MODEL_DIR, shard_file)
        import torch
        with safe_open(shard_path, framework="torch") as f:
            self.embed_weights = f.get_tensor(embed_key).to(torch.float16).numpy()
        print(f"  Embeddings: {self.embed_weights.shape} ({self.embed_weights.nbytes / 1e6:.1f} MB)")

        # ANE chunks
        print("[Hybrid] Loading ANE chunks...")
        self.chunk_0 = ct.models.MLModel(
            os.path.join(HYBRID_DIR, "hybrid_chunk_0_first.mlpackage"),
            compute_units=compute_unit)
        print("  chunk_0 loaded")

        self.mid_chunks = []
        for i in range(1, 8):
            m = ct.models.MLModel(
                os.path.join(HYBRID_DIR, f"hybrid_chunk_{i}_mid.mlpackage"),
                compute_units=compute_unit)
            self.mid_chunks.append(m)
            print(f"  chunk_{i} loaded")

        self.final_chunk = ct.models.MLModel(
            os.path.join(HYBRID_DIR, "hybrid_chunk_final.mlpackage"),
            compute_units=compute_unit)
        print("  chunk_final loaded")

        # LM head
        print("[Hybrid] Loading LM head...")
        self.lm_head = ct.models.MLModel(LM_HEAD_PATH, compute_units=compute_unit)
        print("  lm_head loaded")

        elapsed = time.time() - t0
        print(f"[Hybrid] All models loaded in {elapsed:.1f}s")

    def _reset_state(self):
        """Create fresh decode state."""
        conv_states = [np.zeros((N_DN_PER_BLOCK, LIN_QKV_DIM, CONV_KERNEL), dtype=np.float16) for _ in range(8)]
        ssm_states = [np.zeros((N_DN_PER_BLOCK, LIN_NUM_VAL_HEADS, LIN_KEY_DIM, LIN_VAL_DIM), dtype=np.float16) for _ in range(8)]
        kv_caches = [(
            np.zeros((NUM_KV_HEADS, CONTEXT, HEAD_DIM), dtype=np.float32),
            np.zeros((NUM_KV_HEADS, CONTEXT, HEAD_DIM), dtype=np.float32),
        ) for _ in range(8)]
        return conv_states, ssm_states, kv_caches

    def _embed(self, token_id):
        """CPU embedding lookup. Returns [1, HIDDEN] fp16."""
        return self.embed_weights[token_id:token_id+1]

    def _decode_step(self, x, pos, conv_states, ssm_states, kv_caches):
        """Run one full decode step through all chunks + CPU attention.

        x: [1, HIDDEN] fp16
        Returns: hidden_states [1, HIDDEN] fp16, updated states
        """
        # Chunk 0
        out = self.chunk_0.predict({
            "x": x,
            "conv_states": conv_states[0],
            "ssm_states": ssm_states[0],
        })
        conv_states[0] = out["new_conv_states"]
        ssm_states[0] = out["new_ssm_states"]

        attn_out = cpu_attention(out["q_gate"], out["k"], out["v"], pos, kv_caches, 0)
        residual = out["residual"]

        # Mid chunks 1-7
        for ci in range(7):
            out = self.mid_chunks[ci].predict({
                "attn_output": attn_out,
                "prev_residual": residual,
                "conv_states": conv_states[ci + 1],
                "ssm_states": ssm_states[ci + 1],
            })
            conv_states[ci + 1] = out["new_conv_states"]
            ssm_states[ci + 1] = out["new_ssm_states"]

            attn_out = cpu_attention(out["q_gate"], out["k"], out["v"], pos, kv_caches, ci + 1)
            residual = out["residual"]

        # Final chunk
        out = self.final_chunk.predict({
            "attn_output": attn_out,
            "prev_residual": residual,
        })
        return out["hidden_states"]

    def _sample(self, hidden_states, temperature=0.0, top_p=0.9):
        """Run LM head and sample next token."""
        lm_out = self.lm_head.predict({"input": hidden_states})
        logits = lm_out["output"].flatten().astype(np.float32)

        if temperature <= 0:
            return int(np.argmax(logits))

        logits /= temperature

        # Top-p sampling
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        sorted_logits -= sorted_logits[0]  # stability
        probs = np.exp(sorted_logits)
        probs /= probs.sum()
        cumulative = np.cumsum(probs)
        cutoff = np.searchsorted(cumulative, top_p) + 1
        top_indices = sorted_indices[:cutoff]
        top_probs = probs[:cutoff]
        top_probs /= top_probs.sum()
        return int(np.random.choice(top_indices, p=top_probs))

    def _chat_tokenize(self, messages, enable_thinking=True):
        """Tokenize chat messages using the chat template."""
        try:
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                enable_thinking=enable_thinking, tokenize=False)
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.0, top_p: float = 0.9,
                 raw: bool = False, stream: bool = False):
        """Generate text. Returns generator if stream=True, else string."""
        if raw:
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        else:
            input_ids = self._chat_tokenize([{"role": "user", "content": prompt}])

        if len(input_ids) >= CONTEXT - 2:
            input_ids = input_ids[:CONTEXT - max_tokens - 2]

        gen = self._generate_tokens(input_ids, max_tokens, temperature, top_p)
        if stream:
            return self._stream_text(gen)
        else:
            output_tokens = list(gen)
            return self.tokenizer.decode(output_tokens, skip_special_tokens=True)

    def _generate_tokens(self, input_ids, max_tokens, temperature, top_p):
        """Generator that yields token IDs."""
        conv_states, ssm_states, kv_caches = self._reset_state()

        # Prefill: process all prompt tokens
        for pos, token_id in enumerate(input_ids):
            x = self._embed(token_id)
            hidden = self._decode_step(x, pos, conv_states, ssm_states, kv_caches)

        # Decode
        pos = len(input_ids) - 1
        for _ in range(max_tokens):
            next_token = self._sample(hidden, temperature, top_p)

            if next_token in self.stop_token_ids:
                break

            yield next_token
            pos += 1
            if pos >= CONTEXT - 1:
                break

            x = self._embed(next_token)
            hidden = self._decode_step(x, pos, conv_states, ssm_states, kv_caches)

    def _stream_text(self, token_gen):
        """Yield text chunks as tokens are generated."""
        buffer = []
        for token_id in token_gen:
            buffer.append(token_id)
            text = self.tokenizer.decode(buffer, skip_special_tokens=True)
            if text:
                yield text
                buffer = []
        # Flush remaining
        if buffer:
            text = self.tokenizer.decode(buffer, skip_special_tokens=True)
            if text:
                yield text


# ============ HTTP Server ============

def create_app(model: HybridModel):
    """Create a simple HTTP server using http.server."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import threading

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress request logs

        def do_GET(self):
            if self.path == "/health":
                self._json_response({"status": "ok", "model": "Qwen3.5-4B-hybrid"})
            elif self.path == "/v1/models":
                self._json_response({
                    "data": [{"id": "qwen3.5-4b-hybrid-ane", "object": "model"}]
                })
            else:
                self.send_error(404)

        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length > 0 else {}

            if self.path == "/v1/chat/completions":
                self._handle_chat(body)
            elif self.path == "/v1/completions":
                self._handle_completion(body)
            else:
                self.send_error(404)

        def _handle_chat(self, body):
            messages = body.get("messages", [])
            stream = body.get("stream", False)
            max_tokens = body.get("max_tokens", 256)
            temperature = body.get("temperature", 0.0)
            top_p = body.get("top_p", 0.9)

            input_ids = model._chat_tokenize(messages)

            if len(input_ids) >= CONTEXT - 2:
                input_ids = input_ids[:CONTEXT - max_tokens - 2]

            if stream:
                self._stream_chat(input_ids, max_tokens, temperature, top_p)
            else:
                gen = model._generate_tokens(input_ids, max_tokens, temperature, top_p)
                t0 = time.time()
                tokens = list(gen)
                elapsed = time.time() - t0
                text = model.tokenizer.decode(tokens, skip_special_tokens=True)
                self._json_response({
                    "id": f"chatcmpl-{int(time.time()*1000)}",
                    "object": "chat.completion",
                    "model": "qwen3.5-4b-hybrid-ane",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": len(input_ids),
                        "completion_tokens": len(tokens),
                        "total_tokens": len(input_ids) + len(tokens),
                    }
                })

        def _stream_chat(self, input_ids, max_tokens, temperature, top_p):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            gen = model._generate_tokens(input_ids, max_tokens, temperature, top_p)
            t0 = time.time()
            n_tokens = 0
            buffer = []

            for token_id in gen:
                n_tokens += 1
                buffer.append(token_id)
                text = model.tokenizer.decode(buffer, skip_special_tokens=True)
                if text:
                    chunk = {
                        "id": f"chatcmpl-{int(time.time()*1000)}",
                        "object": "chat.completion.chunk",
                        "model": "qwen3.5-4b-hybrid-ane",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None
                        }]
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    buffer = []

            # Flush remaining buffer
            if buffer:
                text = model.tokenizer.decode(buffer, skip_special_tokens=True)
                if text:
                    chunk = {
                        "id": f"chatcmpl-{int(time.time()*1000)}",
                        "object": "chat.completion.chunk",
                        "model": "qwen3.5-4b-hybrid-ane",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None
                        }]
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()

            # Final chunk
            elapsed = time.time() - t0
            tps = n_tokens / elapsed if elapsed > 0 else 0
            final = {
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion.chunk",
                "model": "qwen3.5-4b-hybrid-ane",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            print(f"[Hybrid] Generated {n_tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")

        def _handle_completion(self, body):
            prompt = body.get("prompt", "")
            max_tokens = body.get("max_tokens", 256)
            temperature = body.get("temperature", 0.0)
            top_p = body.get("top_p", 0.9)

            text = model.generate(prompt, max_tokens=max_tokens,
                                  temperature=temperature, top_p=top_p, raw=True)
            self._json_response({
                "id": f"cmpl-{int(time.time()*1000)}",
                "object": "text_completion",
                "model": "qwen3.5-4b-hybrid-ane",
                "choices": [{"text": text, "finish_reason": "stop"}]
            })

        def _json_response(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve_socket(model: HybridModel, socket_path: str):
    """Run as Unix socket daemon with JSON-lines protocol.

    Compatible with ane_daemon_client.py — same protocol as C++ ane-lm serve.
    """
    import socket
    import signal

    # Clean up stale socket
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(socket_path)
    sock.listen(4)
    sock.settimeout(1.0)  # 1s select timeout for signal handling

    running = True

    def _shutdown(signum, frame):
        nonlocal running
        print(f"\n[Hybrid] Signal {signum}, shutting down...", flush=True)
        running = False

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)

    print(f"[Hybrid] Daemon ready on {socket_path}", flush=True)

    while running:
        try:
            conn, _ = sock.accept()
        except socket.timeout:
            continue
        except OSError:
            break

        conn.settimeout(10.0)
        try:
            # Read one JSON line (max 64KB)
            data = b""
            while b"\n" not in data and len(data) < 65536:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk

            if not data.strip():
                conn.close()
                continue

            req = json.loads(data.strip())
            req_id = req.get("id", "")
            prompt = req.get("prompt", "")
            max_tokens = req.get("max_tokens", 200)
            temperature = req.get("temperature", 0.1)

            t0 = time.time()
            text = model.generate(prompt, max_tokens=max_tokens,
                                  temperature=temperature, raw=True)
            elapsed = time.time() - t0

            # Count tokens in output
            gen_tokens = len(model.tokenizer.encode(text, add_special_tokens=False)) if text else 0
            gen_tps = gen_tokens / elapsed if elapsed > 0 else 0

            resp = json.dumps({
                "id": req_id,
                "text": text or "",
                "gen_tps": round(gen_tps, 1),
                "gen_tokens": gen_tokens,
            }) + "\n"

            # Write full response (handle partial writes)
            resp_bytes = resp.encode()
            sent = 0
            while sent < len(resp_bytes):
                try:
                    n = conn.send(resp_bytes[sent:])
                    if n == 0:
                        break
                    sent += n
                except (BrokenPipeError, ConnectionResetError):
                    break

            print(f"[Hybrid] {gen_tokens} tok in {elapsed:.2f}s ({gen_tps:.1f} tok/s)", flush=True)

        except Exception as e:
            # Send error response
            try:
                err = json.dumps({"id": req.get("id", "") if 'req' in dir() else "",
                                  "text": "", "error": str(e)}) + "\n"
                conn.send(err.encode())
            except Exception:
                pass
            print(f"[Hybrid] Error: {e}", flush=True)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    sock.close()
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    print("[Hybrid] Daemon stopped.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Hybrid ANE/CPU inference server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--serve", action="store_true",
                        help="Run as Unix socket daemon (JSON-lines protocol)")
    parser.add_argument("--socket", default="/tmp/z01_ane_hybrid_4b.sock",
                        help="Unix socket path (for --serve mode)")
    args = parser.parse_args()

    model = HybridModel()
    model.load()

    # Warmup
    print("[Hybrid] Warming up (3 iterations)...")
    for _ in range(3):
        model.generate("Hello", max_tokens=1, raw=True)
    print("[Hybrid] Warmup complete.")

    if args.serve:
        serve_socket(model, args.socket)
    else:
        from http.server import HTTPServer
        Handler = create_app(model)
        server = HTTPServer((args.host, args.port), Handler)
        print(f"\n[Hybrid] Server ready at http://{args.host}:{args.port}")
        print(f"  POST /v1/chat/completions  — OpenAI-compatible chat")
        print(f"  POST /v1/completions       — Raw text completion")
        print(f"  GET  /health               — Health check")
        print(f"  GET  /v1/models            — Model list")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[Hybrid] Shutting down...")
            server.shutdown()


if __name__ == "__main__":
    main()
