#!/usr/bin/env python3
"""Export Qwen3.5 as fused multi-layer CoreML decoder chunks.

Instead of 96 separate CoreML models (3 per layer × 32), this exports 2-4 large
fused chunks where each chunk contains multiple complete transformer layers
(projections + attention/SSM + FFN) as a single CoreML ML Program.

This reduces per-token CoreML calls from 96 to 2-4, eliminating call overhead.

Architecture per layer:
  DeltaNet layers (24/32): norm → first_proj → conv1d → SSM → norm_gate → o_proj → residual → norm → FFN → residual
  Full attn layers (8/32):  norm → QKV_proj → RoPE → GQA attention → o_proj → residual → norm → FFN → residual

For ANE, the key ops that MUST run on ANE (matmul-heavy):
  - first_proj (QKV projection): [2560] → [12288] or [10240+64]
  - o_proj: [4096] → [2560]
  - FFN gate+up+down: [2560] → [9216] → [2560]

Ops that can run on CPU (element-wise / small):
  - RMSNorm, conv1d update, SSM step, RoPE, softmax, residual add
  - These are cheap and run fine on CPU between ANE matmuls

Strategy: Build a PyTorch module that contains ALL ops for N layers,
then trace and convert to CoreML. CoreML's compiler will fuse the
matmuls for ANE and keep element-wise ops on CPU automatically.
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


# ============ Model Components ============

class RMSNorm(nn.Module):
    """Qwen3_5-style RMSNorm: weight applied as (1 + weight), initialized to zeros."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * (1.0 + self.weight)


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeltaNetLayer(nn.Module):
    """Single DeltaNet (linear attention) layer with all ops fused."""
    def __init__(self, hidden_size, intermediate_size, lin_qkv_dim, lin_total_val,
                 lin_num_val_heads, lin_key_dim, lin_val_dim, lin_num_key_heads,
                 conv_kernel=4, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.lin_qkv_dim = lin_qkv_dim  # 2*key_dim + total_val = 2*2048 + 4096 = 8192
        self.lin_total_val = lin_total_val  # 32 * 128 = 4096
        self.lin_num_val_heads = lin_num_val_heads  # 32
        self.lin_key_dim = lin_key_dim  # 128
        self.lin_val_dim = lin_val_dim  # 128
        self.lin_num_key_heads = lin_num_key_heads  # 16
        self.conv_kernel = conv_kernel
        self.head_ratio = lin_num_val_heads // lin_num_key_heads  # 2

        # Pre-attention norm
        self.input_layernorm = RMSNorm(hidden_size, eps)

        # Projections
        ab_dim = 2 * lin_num_val_heads  # 64 (in_proj_a + in_proj_b)
        self.in_proj = nn.Linear(hidden_size, lin_qkv_dim + lin_total_val + ab_dim, bias=False)
        self.out_proj = nn.Linear(lin_total_val, hidden_size, bias=False)

        # Conv1d (depthwise) — stored as flat weight for manual conv
        self.conv_weight = nn.Parameter(torch.zeros(lin_qkv_dim, conv_kernel))

        # SSM parameters
        self.A_log = nn.Parameter(torch.zeros(lin_num_val_heads))
        self.dt_bias = nn.Parameter(torch.zeros(lin_num_val_heads))

        # Output gated norm
        self.norm_weight = nn.Parameter(torch.ones(lin_val_dim))

        # Post-attention norm + FFN
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.ffn = SwiGLUFFN(hidden_size, intermediate_size)

    def forward(self, x, conv_state, ssm_state):
        """
        x: [1, hidden_size]
        conv_state: [lin_qkv_dim, conv_kernel]  (depthwise conv buffer)
        ssm_state: [num_val_heads, key_dim, val_dim]  (recurrent state)
        Returns: (x_out, new_conv_state, new_ssm_state)
        """
        residual = x

        # Pre-attention norm
        h = self.input_layernorm(x)

        # Combined projection: [qkv, z, a, b]
        proj = self.in_proj(h)  # [1, qkv_dim + total_val + 64]

        qkv_dim = self.lin_qkv_dim  # 8192
        mixed_qkv = proj[:, :qkv_dim]  # [1, 8192]
        z = proj[:, qkv_dim:qkv_dim + self.lin_total_val]  # [1, 4096]
        ab = proj[:, qkv_dim + self.lin_total_val:]  # [1, 64]
        a_raw = ab[:, :self.lin_num_val_heads]  # [1, 32]
        b_raw = ab[:, self.lin_num_val_heads:]  # [1, 32]

        # Causal depthwise conv1d (shift state, apply conv)
        # Shift: drop oldest, append new
        new_conv_state = torch.cat([conv_state[:, 1:], mixed_qkv.squeeze(0).unsqueeze(1)], dim=1)
        # Apply conv: sum(state * weight) per channel
        conv_out = (new_conv_state * self.conv_weight).sum(dim=1)  # [qkv_dim]
        conv_out = F.silu(conv_out)

        # Split into Q, K, V
        key_total = self.lin_num_key_heads * self.lin_key_dim  # 2048
        q = conv_out[:key_total]  # [2048]
        k = conv_out[key_total:2*key_total]  # [2048]
        v = conv_out[2*key_total:]  # [4096]

        # Reshape and expand heads
        q = q.view(self.lin_num_key_heads, self.lin_key_dim)  # [16, 128]
        k = k.view(self.lin_num_key_heads, self.lin_key_dim)  # [16, 128]
        v = v.view(self.lin_num_val_heads, self.lin_val_dim)  # [32, 128]

        # Repeat Q, K to match value heads
        q = q.repeat_interleave(self.head_ratio, dim=0)  # [32, 128]
        k = k.repeat_interleave(self.head_ratio, dim=0)  # [32, 128]

        # L2 normalize Q, K
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Compute gating signals
        beta = torch.sigmoid(b_raw).squeeze(0)  # [32]
        g = (-torch.exp(self.A_log) * F.softplus(a_raw.squeeze(0) + self.dt_bias))  # [32]
        decay = torch.exp(g)  # [32]

        # Gated delta rule SSM step (batched across heads)
        # state: [H, key_dim, val_dim], k: [H, key_dim], v: [H, val_dim]
        # Sk = einsum('hkv,hk->hv', state, k)
        Sk = torch.einsum('hkv,hk->hv', ssm_state, k)  # [H, val_dim]
        delta = v - Sk  # [H, val_dim]
        # outer products: k[:,None] * delta[None,:] → [H, key_dim, val_dim]
        outer = k.unsqueeze(2) * delta.unsqueeze(1)  # [H, key_dim, val_dim]
        new_ssm_state = decay.view(-1, 1, 1) * ssm_state + beta.view(-1, 1, 1) * outer
        # y = einsum('hkv,hk->hv', new_state, q)
        y = torch.einsum('hkv,hk->hv', new_ssm_state, q)  # [H, val_dim]

        # Gated RMSNorm on output (batched across heads)
        # Per-head RMSNorm: normalize each head independently
        var = y.pow(2).mean(dim=-1, keepdim=True)  # [H, 1]
        y_norm = y * torch.rsqrt(var + 1e-6) * self.norm_weight.unsqueeze(0)
        # SiLU gate
        z_reshaped = z.view(self.lin_num_val_heads, self.lin_val_dim)  # [H, val_dim]
        y_normed = (y_norm * F.silu(z_reshaped)).reshape(-1)  # [total_val]

        # Output projection
        attn_out = self.out_proj(y_normed.unsqueeze(0))  # [1, hidden_size]

        # Residual 1
        x = residual + attn_out

        # FFN with norm
        residual2 = x
        x = residual2 + self.ffn(self.post_attention_layernorm(x))

        return x, new_conv_state, new_ssm_state


class FullAttentionLayer(nn.Module):
    """Single full attention layer — projections only (attention done externally)."""
    def __init__(self, hidden_size, intermediate_size, num_heads, num_kv_heads,
                 head_dim, attn_output_gate=True, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.attn_output_gate = attn_output_gate

        q_out = num_heads * head_dim * (2 if attn_output_gate else 1)
        kv_out = num_kv_heads * head_dim

        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.q_proj = nn.Linear(hidden_size, q_out, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_out, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_out, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = nn.Parameter(torch.zeros(head_dim))
        self.k_norm = nn.Parameter(torch.zeros(head_dim))

        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.ffn = SwiGLUFFN(hidden_size, intermediate_size)

    def forward_proj(self, x):
        """Just the projections — attention core handled outside CoreML."""
        h = self.input_layernorm(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        return q, k, v

    def forward_post_attn(self, x, attn_output):
        """After attention: o_proj + residual + FFN."""
        out = self.o_proj(attn_output)
        x = x + out
        x = x + self.ffn(self.post_attention_layernorm(x))
        return x


class FusedDecoderChunk(nn.Module):
    """Multiple consecutive layers fused into a single module.

    DeltaNet layers are fully fused (including SSM).
    Full attention layers export projections — attention core done on CPU.
    """
    def __init__(self, layers, layer_types):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.layer_types = layer_types
        self.n_deltanet = sum(1 for t in layer_types if t == "linear_attention")
        self.n_full_attn = sum(1 for t in layer_types if t == "full_attention")

    def forward(self, x, conv_states, ssm_states):
        """
        x: [1, hidden_size]
        conv_states: [n_deltanet, qkv_dim, conv_kernel]
        ssm_states: [n_deltanet, num_val_heads, key_dim, val_dim]
        Returns: (x, new_conv_states, new_ssm_states, full_attn_outputs)
        """
        new_conv = []
        new_ssm = []
        dn_idx = 0

        for i, (layer, lt) in enumerate(zip(self.layers, self.layer_types)):
            if lt == "linear_attention":
                x, cs, ss = layer(x, conv_states[dn_idx], ssm_states[dn_idx])
                new_conv.append(cs)
                new_ssm.append(ss)
                dn_idx += 1
            else:
                # Full attention: can't fuse attention core (needs KV cache)
                # Just do projections + FFN, attention handled outside
                # For now, skip full attention layers in the fused path
                pass

        return x, torch.stack(new_conv), torch.stack(new_ssm)


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
        test_key = f"{pfx}embed_tokens.weight"
        if test_key in weight_map:
            return pfx
    # Try with layers prefix
    for pfx in ["model.language_model.", "language_model.model.", "model.", ""]:
        test_key = f"{pfx}layers.0.input_layernorm.weight"
        if test_key in weight_map:
            return pfx
    raise ValueError("Cannot detect weight prefix")


def load_deltanet_layer(handles, weight_map, prefix, layer_idx, config):
    """Load weights into a DeltaNetLayer."""
    pfx = f"{prefix}layers.{layer_idx}"
    hidden = config["hidden_size"]
    inter = config["intermediate_size"]
    lin_num_key_heads = config["linear_num_key_heads"]
    lin_key_dim = config["linear_key_head_dim"]
    lin_num_val_heads = config["linear_num_value_heads"]
    lin_val_dim = config["linear_value_head_dim"]
    lin_qkv_dim = 2 * lin_num_key_heads * lin_key_dim + lin_num_val_heads * lin_val_dim
    lin_total_val = lin_num_val_heads * lin_val_dim
    conv_kernel = config.get("linear_conv_kernel_dim", 4)
    eps = config.get("rms_norm_eps", 1e-6)

    layer = DeltaNetLayer(
        hidden, inter, lin_qkv_dim, lin_total_val,
        lin_num_val_heads, lin_key_dim, lin_val_dim, lin_num_key_heads,
        conv_kernel, eps
    )

    # Load weights
    with torch.no_grad():
        layer.input_layernorm.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.input_layernorm.weight"))

        # Fuse QKV + Z + A + B into single projection
        qkv_w = get_tensor(handles, weight_map, f"{pfx}.linear_attn.in_proj_qkv.weight")
        z_w = get_tensor(handles, weight_map, f"{pfx}.linear_attn.in_proj_z.weight")
        a_w = get_tensor(handles, weight_map, f"{pfx}.linear_attn.in_proj_a.weight")
        b_w = get_tensor(handles, weight_map, f"{pfx}.linear_attn.in_proj_b.weight")
        fused_w = torch.cat([qkv_w, z_w, a_w, b_w], dim=0)
        layer.in_proj.weight.copy_(fused_w)

        # Conv1d weight: stored as [channels, 1, kernel] → [channels, kernel]
        conv_w = get_tensor(handles, weight_map, f"{pfx}.linear_attn.conv1d.weight")
        layer.conv_weight.copy_(conv_w.squeeze(1))

        # SSM params
        layer.A_log.copy_(get_tensor(handles, weight_map, f"{pfx}.linear_attn.A_log"))
        layer.dt_bias.copy_(get_tensor(handles, weight_map, f"{pfx}.linear_attn.dt_bias"))

        # Output norm
        layer.norm_weight.copy_(get_tensor(handles, weight_map, f"{pfx}.linear_attn.norm.weight"))

        # Output projection
        layer.out_proj.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.linear_attn.out_proj.weight"))

        # Post-attention norm
        layer.post_attention_layernorm.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.post_attention_layernorm.weight"))

        # FFN
        layer.ffn.gate_proj.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.mlp.gate_proj.weight"))
        layer.ffn.up_proj.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.mlp.up_proj.weight"))
        layer.ffn.down_proj.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.mlp.down_proj.weight"))

    return layer


def main():
    parser = argparse.ArgumentParser(description="Export fused Qwen3.5 decoder chunks")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=8, help="Layers per chunk (default: 8)")
    parser.add_argument("--no-quantize", action="store_true", help="Skip LUT6 quantization")
    parser.add_argument("--deltanet-only", action="store_true", help="Only export DeltaNet layers (skip full attention)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load config
    config_path = os.path.join(args.model, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    tc = config.get("text_config", config)

    hidden = tc["hidden_size"]
    num_layers = tc["num_hidden_layers"]
    full_attn_interval = tc.get("full_attention_interval", 4)

    print(f"Model: {args.model}")
    print(f"Hidden: {hidden}, Layers: {num_layers}, Chunk size: {args.chunk_size}")

    # Determine layer types
    layer_types = []
    for i in range(num_layers):
        if (i + 1) % full_attn_interval == 0:
            layer_types.append("full_attention")
        else:
            layer_types.append("linear_attention")

    dn_count = sum(1 for t in layer_types if t == "linear_attention")
    fa_count = sum(1 for t in layer_types if t == "full_attention")
    print(f"DeltaNet layers: {dn_count}, Full attention: {fa_count}")

    # Open weights
    handles, weight_map = open_safetensors(args.model)
    prefix = detect_prefix(handles, weight_map)
    print(f"Weight prefix: '{prefix}'")

    # For now, export DeltaNet layers only as fused chunks
    # Full attention layers need KV cache which can't be easily fused
    print(f"\n--- Exporting DeltaNet layers as fused chunks ---")

    dn_layers = []
    dn_indices = []
    for i in range(num_layers):
        if layer_types[i] == "linear_attention":
            print(f"  Loading DeltaNet layer {i}...", end=" ", flush=True)
            layer = load_deltanet_layer(handles, weight_map, prefix, i, tc)
            dn_layers.append(layer)
            dn_indices.append(i)
            print("OK")

    # Export individual DeltaNet layers first (for testing)
    print(f"\nExporting first DeltaNet layer as test...")
    test_layer = dn_layers[0]
    test_layer.eval()

    lin_qkv_dim = test_layer.lin_qkv_dim
    lin_num_val_heads = test_layer.lin_num_val_heads
    lin_key_dim = test_layer.lin_key_dim
    lin_val_dim = test_layer.lin_val_dim
    conv_kernel = test_layer.conv_kernel

    # Create example inputs
    x = torch.randn(1, hidden)
    conv_state = torch.zeros(lin_qkv_dim, conv_kernel)
    ssm_state = torch.zeros(lin_num_val_heads, lin_key_dim, lin_val_dim)

    # Trace
    print("  Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(test_layer, (x, conv_state, ssm_state))

    # Convert to CoreML
    print("  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x", shape=(1, hidden), dtype=np.float16),
            ct.TensorType(name="conv_state", shape=(lin_qkv_dim, conv_kernel), dtype=np.float16),
            ct.TensorType(name="ssm_state", shape=(lin_num_val_heads, lin_key_dim, lin_val_dim), dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="x_out", dtype=np.float16),
            ct.TensorType(name="new_conv_state", dtype=np.float16),
            ct.TensorType(name="new_ssm_state", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )

    # Quantize
    if not args.no_quantize:
        print("  Applying LUT6...")
        mlmodel = palettize_weights(mlmodel, LUT6_CONFIG)

    # Save
    out_path = os.path.join(args.output, "deltanet_layer_0_test.mlpackage")
    mlmodel.save(out_path)
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(out_path) for f in fns
    ) / 1e6
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")

    # Quick benchmark
    print("  Benchmarking single layer...")
    m = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    inp = {
        "x": np.random.randn(1, hidden).astype(np.float16),
        "conv_state": np.zeros((lin_qkv_dim, conv_kernel), dtype=np.float16),
        "ssm_state": np.zeros((lin_num_val_heads, lin_key_dim, lin_val_dim), dtype=np.float16),
    }
    for _ in range(5):
        m.predict(inp)
    t0 = time.perf_counter()
    N = 30
    for _ in range(N):
        m.predict(inp)
    avg_ms = (time.perf_counter() - t0) / N * 1000
    print(f"  Single DeltaNet layer: {avg_ms:.2f} ms/call")

    # Now export multi-layer fused chunks
    chunk_size = min(args.chunk_size, len(dn_layers))
    n_chunks = (len(dn_layers) + chunk_size - 1) // chunk_size
    print(f"\n--- Exporting {n_chunks} fused chunks of {chunk_size} DeltaNet layers ---")

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, len(dn_layers))
        n_in_chunk = end - start
        chunk_layers = dn_layers[start:end]
        chunk_indices = dn_indices[start:end]

        print(f"\nChunk {ci}: layers {chunk_indices} ({n_in_chunk} layers)")

        # Build fused module
        class MultiDeltaNet(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = nn.ModuleList(layers)
                self.n = len(layers)

            def forward(self, x, conv_states, ssm_states):
                new_convs = []
                new_ssms = []
                for i, layer in enumerate(self.layers):
                    x, cs, ss = layer(x, conv_states[i], ssm_states[i])
                    new_convs.append(cs)
                    new_ssms.append(ss)
                return x, torch.stack(new_convs), torch.stack(new_ssms)

        fused = MultiDeltaNet(chunk_layers)
        fused.eval()

        # Trace
        print("  Tracing...", end=" ", flush=True)
        x_ex = torch.randn(1, hidden)
        cs_ex = torch.zeros(n_in_chunk, lin_qkv_dim, conv_kernel)
        ss_ex = torch.zeros(n_in_chunk, lin_num_val_heads, lin_key_dim, lin_val_dim)
        with torch.no_grad():
            traced = torch.jit.trace(fused, (x_ex, cs_ex, ss_ex))
        print("OK")

        # Convert
        print("  Converting...", end=" ", flush=True)
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="x", shape=(1, hidden), dtype=np.float16),
                ct.TensorType(name="conv_states", shape=(n_in_chunk, lin_qkv_dim, conv_kernel), dtype=np.float16),
                ct.TensorType(name="ssm_states", shape=(n_in_chunk, lin_num_val_heads, lin_key_dim, lin_val_dim), dtype=np.float16),
            ],
            outputs=[
                ct.TensorType(name="x_out", dtype=np.float16),
                ct.TensorType(name="new_conv_states", dtype=np.float16),
                ct.TensorType(name="new_ssm_states", dtype=np.float16),
            ],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
        )
        print("OK")

        # Quantize
        if not args.no_quantize:
            print("  LUT6...", end=" ", flush=True)
            mlmodel = palettize_weights(mlmodel, LUT6_CONFIG)
            print("OK")

        # Save
        chunk_path = os.path.join(args.output, f"deltanet_chunk_{ci}_{n_in_chunk}L.mlpackage")
        mlmodel.save(chunk_path)
        size_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(chunk_path) for f in fns
        ) / 1e6
        print(f"  Saved: {chunk_path} ({size_mb:.1f} MB)")

        # Benchmark
        print("  Benchmarking...", end=" ", flush=True)
        m = ct.models.MLModel(chunk_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        inp = {
            "x": np.random.randn(1, hidden).astype(np.float16),
            "conv_states": np.zeros((n_in_chunk, lin_qkv_dim, conv_kernel), dtype=np.float16),
            "ssm_states": np.zeros((n_in_chunk, lin_num_val_heads, lin_key_dim, lin_val_dim), dtype=np.float16),
        }
        for _ in range(5):
            m.predict(inp)
        t0 = time.perf_counter()
        for _ in range(N):
            m.predict(inp)
        chunk_ms = (time.perf_counter() - t0) / N * 1000
        per_layer = chunk_ms / n_in_chunk
        print(f"{chunk_ms:.2f} ms ({per_layer:.2f} ms/layer)")

    print(f"\nDone! {n_chunks} chunks exported.")


if __name__ == "__main__":
    main()
