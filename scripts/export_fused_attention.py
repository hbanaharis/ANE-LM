#!/usr/bin/env python3
"""Export Qwen3.5 full attention layers as CoreML models with KV cache as MLState.

Each full attention layer (8 total at positions 3,7,11,15,19,23,27,31) is exported
as a separate CoreML ML Program. KV cache is managed as ct.StateType — CoreML holds
it in-place between calls, eliminating data transfer overhead.

Architecture per full attention layer:
  input_layernorm -> Q/K/V projections -> per-head RMSNorm on Q,K ->
  RoPE (partial, 25%) -> KV cache update -> GQA attention ->
  sigmoid output gate -> o_proj -> residual -> post_attn_norm -> FFN -> residual
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from coremltools.optimize.coreml import (
    palettize_weights,
    OpPalettizerConfig,
    OptimizationConfig,
)
from safetensors import safe_open


LUT6_CONFIG = OptimizationConfig(global_config=OpPalettizerConfig(nbits=6))

CONTEXT_LENGTH = 4096
MODEL_DTYPE = torch.float16


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FullAttentionLayerStateful(nn.Module):
    """Full attention layer with KV cache as registered buffer (for ct.StateType).

    The KV cache is stored as a single buffer of shape:
    [2, num_kv_heads, context_length, head_dim]
    where dim 0 = [K, V].

    CoreML's state system will manage this buffer in-place between calls.
    """

    def __init__(self, hidden_size, intermediate_size, num_heads, num_kv_heads,
                 head_dim, rope_dim, rope_theta, context_length, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.groups = num_heads // num_kv_heads
        self.context_length = context_length

        # Projections — Q is doubled for output gate
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = nn.Parameter(torch.ones(head_dim))
        self.k_norm = nn.Parameter(torch.ones(head_dim))

        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.ffn = SwiGLUFFN(hidden_size, intermediate_size)

        # RoPE cos/sin precomputed
        half_rot = rope_dim // 2
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
        positions = torch.arange(context_length, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer('rope_cos', freqs.cos().to(MODEL_DTYPE))
        self.register_buffer('rope_sin', freqs.sin().to(MODEL_DTYPE))

        # KV cache as registered buffer — this becomes ct.StateType
        self.register_buffer(
            'kv_cache',
            torch.zeros(2, num_kv_heads, context_length, head_dim, dtype=MODEL_DTYPE)
        )

    def _per_head_rmsnorm(self, x, weight, num_heads):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + 1e-6) * weight.unsqueeze(0)

    def _apply_rope(self, x, cos, sin):
        """Apply RoPE to first rope_dim dims. x: [H, head_dim]"""
        half_rot = self.rope_dim // 2
        x_rot = x[:, :self.rope_dim]
        x_pass = x[:, self.rope_dim:]
        x1 = x_rot[:, :half_rot]
        x2 = x_rot[:, half_rot:]
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        return torch.cat([out1, out2, x_pass], dim=-1)

    def forward(self, hidden_states, current_pos, causal_mask):
        """
        hidden_states: [1, 1, hidden_size]
        current_pos: [1] int32 — position for RoPE and KV cache write
        causal_mask: [1, 1, 1, context_length] float16 — 0 for attend, -inf for mask
        Returns: [1, 1, hidden_size]
        """
        residual = hidden_states

        # Pre-attention norm
        h = self.input_layernorm(hidden_states)  # [1, 1, H]

        # Projections
        q_gate = self.q_proj(h).view(self.num_heads, self.head_dim * 2)
        q = q_gate[:, :self.head_dim]
        gate = q_gate[:, self.head_dim:]
        k = self.k_proj(h).view(self.num_kv_heads, self.head_dim)
        v = self.v_proj(h).view(self.num_kv_heads, self.head_dim)

        # Per-head RMSNorm
        q = self._per_head_rmsnorm(q, self.q_norm, self.num_heads)
        k = self._per_head_rmsnorm(k, self.k_norm, self.num_kv_heads)

        # RoPE
        p = current_pos[0]
        cos = self.rope_cos[p]
        sin = self.rope_sin[p]
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # KV cache update — write at current_pos
        # This modifies the registered buffer in-place (CoreML state)
        self.kv_cache[0, :, p:p+1, :] = k.unsqueeze(1)  # K
        self.kv_cache[1, :, p:p+1, :] = v.unsqueeze(1)  # V

        # Read from cache
        k_cache = self.kv_cache[0]  # [num_kv_heads, ctx, head_dim]
        v_cache = self.kv_cache[1]

        # GQA: expand KV heads
        k_expanded = k_cache.unsqueeze(1).expand(-1, self.groups, -1, -1)
        k_expanded = k_expanded.reshape(self.num_heads, self.context_length, self.head_dim)
        v_expanded = v_cache.unsqueeze(1).expand(-1, self.groups, -1, -1)
        v_expanded = v_expanded.reshape(self.num_heads, self.context_length, self.head_dim)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum('hd,hcd->hc', q, k_expanded) * scale  # [H, ctx]

        # Apply causal mask
        scores = scores + causal_mask.view(1, self.context_length)

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.einsum('hc,hcd->hd', attn_weights, v_expanded)  # [H, head_dim]

        # Output gate
        attn_out = attn_out * torch.sigmoid(gate)

        # Output projection
        out = self.o_proj(attn_out.reshape(1, 1, self.num_heads * self.head_dim))

        # Residual + FFN
        x = residual + out
        x = x + self.ffn(self.post_attention_layernorm(x))
        return x


# ============ Weight Loading ============

def open_safetensors(model_dir):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_names = sorted(set(index["weight_map"].values()))
        handles = {
            name: safe_open(os.path.join(model_dir, name), framework="torch")
            for name in shard_names
        }
        return handles, index["weight_map"]
    else:
        files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
        handles = {f: safe_open(os.path.join(model_dir, f), framework="torch") for f in files}
        weight_map = {}
        for name, handle in handles.items():
            for key in handle.keys():
                weight_map[key] = name
        return handles, weight_map


def get_tensor(handles, weight_map, key):
    shard = weight_map[key]
    return handles[shard].get_tensor(key).to(torch.float32)


def detect_prefix(handles, weight_map):
    for pfx in ["model.language_model.", "language_model.model.", "model.", ""]:
        test_key = f"{pfx}layers.0.input_layernorm.weight"
        if test_key in weight_map:
            return pfx
    raise ValueError("Cannot detect weight prefix")


def load_full_attention_layer(handles, weight_map, prefix, layer_idx, config, ctx):
    pfx = f"{prefix}layers.{layer_idx}"
    hidden = config["hidden_size"]
    inter = config["intermediate_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    eps = config.get("rms_norm_eps", 1e-6)

    rope_params = config.get("rope_parameters", {})
    rope_theta = rope_params.get("rope_theta", 10000000.0)
    partial_rotary = rope_params.get("partial_rotary_factor", 0.25)
    rope_dim = int(head_dim * partial_rotary)

    layer = FullAttentionLayerStateful(
        hidden, inter, num_heads, num_kv_heads, head_dim,
        rope_dim, rope_theta, ctx, eps
    )

    with torch.no_grad():
        layer.input_layernorm.weight.copy_(
            get_tensor(handles, weight_map, f"{pfx}.input_layernorm.weight"))
        layer.q_proj.weight.copy_(
            get_tensor(handles, weight_map, f"{pfx}.self_attn.q_proj.weight"))
        layer.k_proj.weight.copy_(
            get_tensor(handles, weight_map, f"{pfx}.self_attn.k_proj.weight"))
        layer.v_proj.weight.copy_(
            get_tensor(handles, weight_map, f"{pfx}.self_attn.v_proj.weight"))
        layer.o_proj.weight.copy_(
            get_tensor(handles, weight_map, f"{pfx}.self_attn.o_proj.weight"))
        layer.q_norm.copy_(
            get_tensor(handles, weight_map, f"{pfx}.self_attn.q_norm.weight"))
        layer.k_norm.copy_(
            get_tensor(handles, weight_map, f"{pfx}.self_attn.k_norm.weight"))
        layer.post_attention_layernorm.weight.copy_(
            get_tensor(handles, weight_map, f"{pfx}.post_attention_layernorm.weight"))
        layer.ffn.gate_proj.weight.copy_(
            get_tensor(handles, weight_map, f"{pfx}.mlp.gate_proj.weight"))
        layer.ffn.up_proj.weight.copy_(
            get_tensor(handles, weight_map, f"{pfx}.mlp.up_proj.weight"))
        layer.ffn.down_proj.weight.copy_(
            get_tensor(handles, weight_map, f"{pfx}.mlp.down_proj.weight"))

    return layer


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3.5 full attention layers with KV cache state")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--context", type=int, default=CONTEXT_LENGTH, help="Context length for KV cache")
    parser.add_argument("--no-quantize", action="store_true", help="Skip LUT6 quantization")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    config_path = os.path.join(args.model, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    tc = config.get("text_config", config)

    hidden = tc["hidden_size"]
    num_layers = tc["num_hidden_layers"]
    num_heads = tc["num_attention_heads"]
    num_kv_heads = tc["num_key_value_heads"]
    head_dim = tc["head_dim"]
    full_attn_interval = tc.get("full_attention_interval", 4)

    attn_indices = [i for i in range(num_layers) if (i + 1) % full_attn_interval == 0]
    print(f"Model: {args.model}")
    print(f"Hidden: {hidden}, Heads: {num_heads}, KV heads: {num_kv_heads}, Head dim: {head_dim}")
    print(f"Full attention layers ({len(attn_indices)}): {attn_indices}")
    print(f"Context length: {args.context}")

    handles, weight_map = open_safetensors(args.model)
    prefix = detect_prefix(handles, weight_map)
    print(f"Weight prefix: '{prefix}'")

    # KV cache state spec for CoreML
    kv_state = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(2, num_kv_heads, args.context, head_dim),
                dtype=np.float16,
            ),
            name="kv_cache",
        )
    ]
    kv_cache_mb = 2 * num_kv_heads * args.context * head_dim * 2 / 1e6
    print(f"KV cache per layer: {kv_cache_mb:.1f} MB (state, FP16)")

    N_BENCH = 30

    for i, layer_idx in enumerate(attn_indices):
        print(f"\nAttention layer {layer_idx} (#{i+1}/{len(attn_indices)}):")

        print("  Loading weights...", end=" ", flush=True)
        layer = load_full_attention_layer(handles, weight_map, prefix, layer_idx, tc, args.context)
        layer = layer.to(MODEL_DTYPE)
        layer.eval()
        print("OK")

        # Example inputs for tracing
        hidden_states = torch.zeros(1, 1, hidden, dtype=MODEL_DTYPE)
        current_pos = torch.zeros(1, dtype=torch.int32)
        causal_mask = torch.zeros(1, 1, 1, args.context, dtype=MODEL_DTYPE)

        # Reset KV cache buffer before tracing
        layer.kv_cache.zero_()

        print("  Tracing...", end=" ", flush=True)
        with torch.no_grad():
            traced = torch.jit.trace(layer, (hidden_states, current_pos, causal_mask))
        # Reset buffer in traced model too
        for name, buf in traced.named_buffers():
            if 'kv_cache' in name:
                buf.zero_()
        print("OK")

        print("  Converting to CoreML...", end=" ", flush=True)
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="hidden_states", shape=(1, 1, hidden), dtype=np.float16),
                ct.TensorType(name="current_pos", shape=(1,), dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=(1, 1, 1, args.context), dtype=np.float16),
            ],
            outputs=[
                ct.TensorType(name="output_hidden_states", dtype=np.float16),
            ],
            states=kv_state,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        print("OK")

        if not args.no_quantize:
            print("  LUT6...", end=" ", flush=True)
            mlmodel = palettize_weights(mlmodel, LUT6_CONFIG)
            print("OK")

        out_path = os.path.join(args.output, f"full_attn_L{layer_idx}_stateful.mlpackage")
        mlmodel.save(out_path)
        size_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(out_path) for f in fns
        ) / 1e6
        print(f"  Saved: {out_path} ({size_mb:.1f} MB)")

        # Benchmark with state
        print("  Benchmarking...", end=" ", flush=True)
        m = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        state = m.make_state()
        inp = {
            "hidden_states": np.random.randn(1, 1, hidden).astype(np.float16),
            "current_pos": np.array([10], dtype=np.int32),
            "causal_mask": np.zeros((1, 1, 1, args.context), dtype=np.float16),
        }
        # Warmup
        for _ in range(10):
            m.predict(inp, state)
        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            m.predict(inp, state)
        avg_ms = (time.perf_counter() - t0) / N_BENCH * 1000
        print(f"{avg_ms:.2f} ms")

    print(f"\nDone! {len(attn_indices)} stateful attention layers exported.")
    total_kv = len(attn_indices) * kv_cache_mb
    print(f"Total KV cache: {total_kv:.1f} MB")


if __name__ == "__main__":
    main()
