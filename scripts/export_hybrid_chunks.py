#!/usr/bin/env python3
"""Export Qwen3.5-4B as hybrid ANE/CPU decode chunks.

Pipeline per token:
  embed → [ANE chunk_0] → [CPU attn_0] → [ANE chunk_1] → [CPU attn_1] → ... → [ANE final] → lm_head

Each ANE "hybrid chunk" fuses:
  post-attn of prev block (o_proj + residual + FFN) +
  3 DeltaNet layers (full, with SSM state) +
  pre-attn of current block (input_norm + QKV proj + per-head norms)

CPU handles only the minimal attention core:
  RoPE + KV cache write + GQA attention + sigmoid gate

This gives exactly 1 ANE-to-CPU boundary per 4-layer super-block.

Chunk types:
  chunk_0 (first):  3 DN layers + pre-attn QKV
  chunk_1..7 (mid):  post-attn + 3 DN layers + pre-attn QKV
  chunk_final:       post-attn + final_norm
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


# ============ Components ============

class RMSNorm(nn.Module):
    """Qwen3_5-style RMSNorm: weight applied as (1 + weight), initialized to zeros.
    Uses [x, -x] → LayerNorm trick so CoreML maps it to ANE's native LayerNorm HW.
    See: https://huggingface.co/blog/anemll/anemll-style-rms-ane
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # Concatenate [x, -x] → symmetric around zero → mean=0
        # LayerNorm on symmetric input ≡ RMSNorm
        doubled = torch.cat([x, -x], dim=-1)
        normed = F.layer_norm(doubled, [doubled.shape[-1]], eps=self.eps)
        # Take first half = x / RMS(x)
        return normed[..., :x.shape[-1]] * (1.0 + self.weight)


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeltaNetLayer(nn.Module):
    """Full DeltaNet layer (same as export_fused_decoder.py)."""
    def __init__(self, hidden_size, intermediate_size, lin_qkv_dim, lin_total_val,
                 lin_num_val_heads, lin_key_dim, lin_val_dim, lin_num_key_heads,
                 conv_kernel=4, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.lin_qkv_dim = lin_qkv_dim
        self.lin_total_val = lin_total_val
        self.lin_num_val_heads = lin_num_val_heads
        self.lin_key_dim = lin_key_dim
        self.lin_val_dim = lin_val_dim
        self.lin_num_key_heads = lin_num_key_heads
        self.conv_kernel = conv_kernel
        self.head_ratio = lin_num_val_heads // lin_num_key_heads

        self.input_layernorm = RMSNorm(hidden_size, eps)
        ab_dim = 2 * lin_num_val_heads
        self.in_proj = nn.Linear(hidden_size, lin_qkv_dim + lin_total_val + ab_dim, bias=False)
        self.out_proj = nn.Linear(lin_total_val, hidden_size, bias=False)
        self.conv_weight = nn.Parameter(torch.zeros(lin_qkv_dim, conv_kernel))
        self.A_log = nn.Parameter(torch.zeros(lin_num_val_heads))
        self.dt_bias = nn.Parameter(torch.zeros(lin_num_val_heads))
        self.norm_weight = nn.Parameter(torch.ones(lin_val_dim))
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.ffn = SwiGLUFFN(hidden_size, intermediate_size)

    def forward(self, x, conv_state, ssm_state):
        residual = x
        h = self.input_layernorm(x)
        proj = self.in_proj(h)
        qkv_dim = self.lin_qkv_dim
        mixed_qkv = proj[:, :qkv_dim]
        z = proj[:, qkv_dim:qkv_dim + self.lin_total_val]
        ab = proj[:, qkv_dim + self.lin_total_val:]
        a_raw = ab[:, :self.lin_num_val_heads]
        b_raw = ab[:, self.lin_num_val_heads:]

        new_conv_state = torch.cat([conv_state[:, 1:], mixed_qkv.squeeze(0).unsqueeze(1)], dim=1)
        conv_out = (new_conv_state * self.conv_weight).sum(dim=1)
        conv_out = F.silu(conv_out)

        key_total = self.lin_num_key_heads * self.lin_key_dim
        q = conv_out[:key_total].view(self.lin_num_key_heads, self.lin_key_dim)
        k = conv_out[key_total:2*key_total].view(self.lin_num_key_heads, self.lin_key_dim)
        v = conv_out[2*key_total:].view(self.lin_num_val_heads, self.lin_val_dim)

        q = q.repeat_interleave(self.head_ratio, dim=0)
        k = k.repeat_interleave(self.head_ratio, dim=0)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Scale query (matches transformers reference)
        scale = 1.0 / (self.lin_key_dim ** 0.5)
        q = q * scale

        beta = torch.sigmoid(b_raw).squeeze(0)
        g = (-torch.exp(self.A_log) * F.softplus(a_raw.squeeze(0) + self.dt_bias))
        decay = torch.exp(g)

        # Gated delta rule (matches transformers torch_recurrent_gated_delta_rule):
        # 1. Decay state first
        # 2. Retrieve from DECAYED state
        # 3. Delta = (v - retrieval) * beta
        # 4. Update state with k ⊗ delta (no extra beta)
        decayed_state = decay.view(-1, 1, 1) * ssm_state
        Sk = torch.einsum('hkv,hk->hv', decayed_state, k)
        delta = (v - Sk) * beta.view(-1, 1)
        outer = k.unsqueeze(2) * delta.unsqueeze(1)
        new_ssm_state = decayed_state + outer
        y = torch.einsum('hkv,hk->hv', new_ssm_state, q)

        # Per-head RMSNorm via [y, -y] → LayerNorm trick (ANE-native)
        doubled_y = torch.cat([y, -y], dim=-1)
        normed_y = F.layer_norm(doubled_y, [doubled_y.shape[-1]], eps=1e-6)
        y_norm = normed_y[..., :y.shape[-1]] * self.norm_weight.unsqueeze(0)
        z_reshaped = z.view(self.lin_num_val_heads, self.lin_val_dim)
        y_normed = (y_norm * F.silu(z_reshaped)).reshape(-1)

        attn_out = self.out_proj(y_normed.unsqueeze(0))
        x = residual + attn_out
        residual2 = x
        x = residual2 + self.ffn(self.post_attention_layernorm(x))
        return x, new_conv_state, new_ssm_state


class PreAttnQKV(nn.Module):
    """Pre-attention: input_norm + Q/K/V projection + per-head norms."""
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.input_layernorm = RMSNorm(hidden_size, eps)
        # Q output is doubled for output gate (attn_output_gate=True)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.q_norm = nn.Parameter(torch.zeros(head_dim))
        self.k_norm = nn.Parameter(torch.zeros(head_dim))

    def _per_head_rmsnorm(self, x, weight):
        """Qwen3_5-style per-head RMSNorm: (1 + weight).
        Uses [x, -x] → LayerNorm trick for ANE-native execution."""
        doubled = torch.cat([x, -x], dim=-1)
        normed = F.layer_norm(doubled, [doubled.shape[-1]], eps=1e-6)
        return normed[..., :x.shape[-1]] * (1.0 + weight).unsqueeze(0)

    def forward(self, x):
        """Returns q_gate [num_heads, head_dim*2], k [kv_heads, head_dim], v [kv_heads, head_dim]."""
        h = self.input_layernorm(x)
        q_gate = self.q_proj(h).view(self.num_heads, self.head_dim * 2)
        k = self.k_proj(h).view(self.num_kv_heads, self.head_dim)
        v = self.v_proj(h).view(self.num_kv_heads, self.head_dim)

        # Per-head RMSNorm on Q (first half only, gate is raw) and K
        q = q_gate[:, :self.head_dim]
        gate = q_gate[:, self.head_dim:]
        q = self._per_head_rmsnorm(q, self.q_norm)
        k = self._per_head_rmsnorm(k, self.k_norm)

        # Concatenate q and gate back for output
        q_gate_normed = torch.cat([q, gate], dim=-1)
        return q_gate_normed, k, v


class PostAttn(nn.Module):
    """Post-attention: o_proj + residual + post_norm + FFN + residual."""
    def __init__(self, hidden_size, intermediate_size, num_heads, head_dim, eps=1e-6):
        super().__init__()
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.ffn = SwiGLUFFN(hidden_size, intermediate_size)

    def forward(self, attn_output, residual):
        """attn_output: [1, num_heads*head_dim], residual: [1, hidden_size]."""
        x = residual + self.o_proj(attn_output)
        x = x + self.ffn(self.post_attention_layernorm(x))
        return x


# ============ Hybrid Chunks ============

class FirstChunk(nn.Module):
    """chunk_0: 3 DeltaNet layers + pre-attn QKV projection."""
    def __init__(self, dn_layers, pre_attn):
        super().__init__()
        self.dn_layers = nn.ModuleList(dn_layers)
        self.pre_attn = pre_attn
        self.n_dn = len(dn_layers)

    def forward(self, x, conv_states, ssm_states):
        """
        x: [1, hidden]
        conv_states: [n_dn, qkv_dim, conv_kernel]
        ssm_states: [n_dn, num_val_heads, key_dim, val_dim]
        Returns: q_gate, k, v, residual, new_conv_states, new_ssm_states
        """
        new_convs = []
        new_ssms = []
        for i, layer in enumerate(self.dn_layers):
            x, cs, ss = layer(x, conv_states[i], ssm_states[i])
            new_convs.append(cs)
            new_ssms.append(ss)

        # Pre-attention projection (save residual for CPU→ANE handoff)
        residual = x
        q_gate, k, v = self.pre_attn(x)

        return q_gate, k, v, residual, torch.stack(new_convs), torch.stack(new_ssms)


class MidChunk(nn.Module):
    """chunk_1..7: post-attn + 3 DeltaNet layers + pre-attn QKV."""
    def __init__(self, post_attn, dn_layers, pre_attn):
        super().__init__()
        self.post_attn = post_attn
        self.dn_layers = nn.ModuleList(dn_layers)
        self.pre_attn = pre_attn
        self.n_dn = len(dn_layers)

    def forward(self, attn_output, prev_residual, conv_states, ssm_states):
        """
        attn_output: [1, num_heads*head_dim] — from CPU attention
        prev_residual: [1, hidden] — saved before previous QKV projection
        conv_states: [n_dn, qkv_dim, conv_kernel]
        ssm_states: [n_dn, num_val_heads, key_dim, val_dim]
        """
        # Post-attention of previous block
        x = self.post_attn(attn_output, prev_residual)

        # DeltaNet layers
        new_convs = []
        new_ssms = []
        for i, layer in enumerate(self.dn_layers):
            x, cs, ss = layer(x, conv_states[i], ssm_states[i])
            new_convs.append(cs)
            new_ssms.append(ss)

        # Pre-attention projection
        residual = x
        q_gate, k, v = self.pre_attn(x)

        return q_gate, k, v, residual, torch.stack(new_convs), torch.stack(new_ssms)


class FinalChunk(nn.Module):
    """Last chunk: post-attn + final_norm."""
    def __init__(self, post_attn, final_norm):
        super().__init__()
        self.post_attn = post_attn
        self.final_norm = final_norm

    def forward(self, attn_output, prev_residual):
        x = self.post_attn(attn_output, prev_residual)
        return self.final_norm(x)


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


def load_deltanet_layer(handles, weight_map, prefix, layer_idx, config):
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

    with torch.no_grad():
        layer.input_layernorm.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.input_layernorm.weight"))
        qkv_w = get_tensor(handles, weight_map, f"{pfx}.linear_attn.in_proj_qkv.weight")
        z_w = get_tensor(handles, weight_map, f"{pfx}.linear_attn.in_proj_z.weight")
        a_w = get_tensor(handles, weight_map, f"{pfx}.linear_attn.in_proj_a.weight")
        b_w = get_tensor(handles, weight_map, f"{pfx}.linear_attn.in_proj_b.weight")
        layer.in_proj.weight.copy_(torch.cat([qkv_w, z_w, a_w, b_w], dim=0))
        conv_w = get_tensor(handles, weight_map, f"{pfx}.linear_attn.conv1d.weight")
        layer.conv_weight.copy_(conv_w.squeeze(1))
        layer.A_log.copy_(get_tensor(handles, weight_map, f"{pfx}.linear_attn.A_log"))
        layer.dt_bias.copy_(get_tensor(handles, weight_map, f"{pfx}.linear_attn.dt_bias"))
        layer.norm_weight.copy_(get_tensor(handles, weight_map, f"{pfx}.linear_attn.norm.weight"))
        layer.out_proj.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.linear_attn.out_proj.weight"))
        layer.post_attention_layernorm.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.post_attention_layernorm.weight"))
        layer.ffn.gate_proj.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.mlp.gate_proj.weight"))
        layer.ffn.up_proj.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.mlp.up_proj.weight"))
        layer.ffn.down_proj.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.mlp.down_proj.weight"))

    return layer


def load_pre_attn(handles, weight_map, prefix, layer_idx, config):
    pfx = f"{prefix}layers.{layer_idx}"
    hidden = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    eps = config.get("rms_norm_eps", 1e-6)

    mod = PreAttnQKV(hidden, num_heads, num_kv_heads, head_dim, eps)
    with torch.no_grad():
        mod.input_layernorm.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.input_layernorm.weight"))
        mod.q_proj.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.self_attn.q_proj.weight"))
        mod.k_proj.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.self_attn.k_proj.weight"))
        mod.v_proj.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.self_attn.v_proj.weight"))
        mod.q_norm.copy_(get_tensor(handles, weight_map, f"{pfx}.self_attn.q_norm.weight"))
        mod.k_norm.copy_(get_tensor(handles, weight_map, f"{pfx}.self_attn.k_norm.weight"))
    return mod


def load_post_attn(handles, weight_map, prefix, layer_idx, config):
    pfx = f"{prefix}layers.{layer_idx}"
    hidden = config["hidden_size"]
    inter = config["intermediate_size"]
    num_heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    eps = config.get("rms_norm_eps", 1e-6)

    mod = PostAttn(hidden, inter, num_heads, head_dim, eps)
    with torch.no_grad():
        mod.o_proj.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.self_attn.o_proj.weight"))
        mod.post_attention_layernorm.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.post_attention_layernorm.weight"))
        mod.ffn.gate_proj.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.mlp.gate_proj.weight"))
        mod.ffn.up_proj.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.mlp.up_proj.weight"))
        mod.ffn.down_proj.weight.copy_(get_tensor(handles, weight_map, f"{pfx}.mlp.down_proj.weight"))
    return mod


class LMHeadArgmax(nn.Module):
    """LM head with in-model top-k: returns (top_k_indices, top_k_logits) instead of full vocab logits.
    Reduces ANE→CPU data transfer from 248K floats to 2*top_k values.
    Inspired by ANEMLL's in-model argmax approach."""
    def __init__(self, hidden_size, vocab_size, topk=32):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)
        self.topk = topk

    def forward(self, x):
        logits = self.proj(x)  # [1, vocab_size]
        values, indices = torch.topk(logits, self.topk, dim=-1)  # [1, topk] each
        return values, indices.to(torch.float32)  # CoreML needs float for indices


def main():
    parser = argparse.ArgumentParser(description="Export hybrid ANE/CPU decode chunks")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--no-quantize", action="store_true", help="Skip LUT6 quantization")
    parser.add_argument("--lm-head", action="store_true", help="Also export LM head with in-model argmax")
    parser.add_argument("--topk", type=int, default=32, help="Top-K for in-model argmax LM head (default: 32)")
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
    lin_num_key_heads = tc["linear_num_key_heads"]
    lin_key_dim = tc["linear_key_head_dim"]
    lin_num_val_heads = tc["linear_num_value_heads"]
    lin_val_dim = tc["linear_value_head_dim"]
    lin_qkv_dim = 2 * lin_num_key_heads * lin_key_dim + lin_num_val_heads * lin_val_dim
    conv_kernel = tc.get("linear_conv_kernel_dim", 4)

    # Build super-block structure
    super_blocks = []
    current_block = {"dn": [], "fa": None}
    for i in range(num_layers):
        if (i + 1) % full_attn_interval == 0:
            current_block["fa"] = i
            super_blocks.append(current_block)
            current_block = {"dn": [], "fa": None}
        else:
            current_block["dn"].append(i)

    n_blocks = len(super_blocks)
    n_dn_per_block = len(super_blocks[0]["dn"])

    print(f"Model: {args.model}")
    print(f"Hidden: {hidden}, Layers: {num_layers}")
    print(f"Super-blocks: {n_blocks}, DN/block: {n_dn_per_block}")
    for i, sb in enumerate(super_blocks):
        print(f"  Block {i}: DN {sb['dn']}, FA {sb['fa']}")

    # Open weights
    handles, weight_map = open_safetensors(args.model)
    prefix = detect_prefix(handles, weight_map)
    print(f"Weight prefix: '{prefix}'")

    N_BENCH = 30
    attn_dim = num_heads * head_dim  # 4096

    # ==================== Chunk 0 (first) ====================
    print(f"\n=== Chunk 0 (first): DN layers {super_blocks[0]['dn']} + pre-attn L{super_blocks[0]['fa']} ===")
    dn_layers_0 = []
    for idx in super_blocks[0]["dn"]:
        print(f"  Loading DN layer {idx}...", end=" ", flush=True)
        dn_layers_0.append(load_deltanet_layer(handles, weight_map, prefix, idx, tc))
        print("OK")
    pre_attn_0 = load_pre_attn(handles, weight_map, prefix, super_blocks[0]["fa"], tc)

    chunk0 = FirstChunk(dn_layers_0, pre_attn_0)
    chunk0.eval()

    # Trace
    x_ex = torch.randn(1, hidden)
    cs_ex = torch.zeros(n_dn_per_block, lin_qkv_dim, conv_kernel)
    ss_ex = torch.zeros(n_dn_per_block, lin_num_val_heads, lin_key_dim, lin_val_dim)

    print("  Tracing...", end=" ", flush=True)
    with torch.no_grad():
        traced = torch.jit.trace(chunk0, (x_ex, cs_ex, ss_ex))
    print("OK")

    print("  Converting...", end=" ", flush=True)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x", shape=(1, hidden), dtype=np.float16),
            ct.TensorType(name="conv_states", shape=(n_dn_per_block, lin_qkv_dim, conv_kernel), dtype=np.float16),
            ct.TensorType(name="ssm_states", shape=(n_dn_per_block, lin_num_val_heads, lin_key_dim, lin_val_dim), dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="q_gate", dtype=np.float16),
            ct.TensorType(name="k", dtype=np.float16),
            ct.TensorType(name="v", dtype=np.float16),
            ct.TensorType(name="residual", dtype=np.float16),
            ct.TensorType(name="new_conv_states", dtype=np.float16),
            ct.TensorType(name="new_ssm_states", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    print("OK")

    if not args.no_quantize:
        print("  LUT6...", end=" ", flush=True)
        mlmodel = palettize_weights(mlmodel, LUT6_CONFIG)
        print("OK")

    out_path = os.path.join(args.output, "hybrid_chunk_0_first.mlpackage")
    mlmodel.save(out_path)
    size_mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(out_path) for f in fns) / 1e6
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")

    # Benchmark chunk 0
    m = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    inp = {
        "x": np.random.randn(1, hidden).astype(np.float16),
        "conv_states": np.zeros((n_dn_per_block, lin_qkv_dim, conv_kernel), dtype=np.float16),
        "ssm_states": np.zeros((n_dn_per_block, lin_num_val_heads, lin_key_dim, lin_val_dim), dtype=np.float16),
    }
    for _ in range(5):
        m.predict(inp)
    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        m.predict(inp)
    avg_ms = (time.perf_counter() - t0) / N_BENCH * 1000
    print(f"  Benchmark: {avg_ms:.2f} ms")

    # ==================== Mid chunks (1..n_blocks-1) ====================
    for bi in range(1, n_blocks):
        sb = super_blocks[bi]
        prev_fa = super_blocks[bi - 1]["fa"]
        print(f"\n=== Chunk {bi} (mid): post-attn L{prev_fa} + DN {sb['dn']} + pre-attn L{sb['fa']} ===")

        post_attn = load_post_attn(handles, weight_map, prefix, prev_fa, tc)
        dn_layers = []
        for idx in sb["dn"]:
            print(f"  Loading DN layer {idx}...", end=" ", flush=True)
            dn_layers.append(load_deltanet_layer(handles, weight_map, prefix, idx, tc))
            print("OK")
        pre_attn = load_pre_attn(handles, weight_map, prefix, sb["fa"], tc)

        chunk = MidChunk(post_attn, dn_layers, pre_attn)
        chunk.eval()

        # Trace
        attn_out_ex = torch.randn(1, attn_dim)
        res_ex = torch.randn(1, hidden)

        print("  Tracing...", end=" ", flush=True)
        with torch.no_grad():
            traced = torch.jit.trace(chunk, (attn_out_ex, res_ex, cs_ex, ss_ex))
        print("OK")

        print("  Converting...", end=" ", flush=True)
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="attn_output", shape=(1, attn_dim), dtype=np.float16),
                ct.TensorType(name="prev_residual", shape=(1, hidden), dtype=np.float16),
                ct.TensorType(name="conv_states", shape=(n_dn_per_block, lin_qkv_dim, conv_kernel), dtype=np.float16),
                ct.TensorType(name="ssm_states", shape=(n_dn_per_block, lin_num_val_heads, lin_key_dim, lin_val_dim), dtype=np.float16),
            ],
            outputs=[
                ct.TensorType(name="q_gate", dtype=np.float16),
                ct.TensorType(name="k", dtype=np.float16),
                ct.TensorType(name="v", dtype=np.float16),
                ct.TensorType(name="residual", dtype=np.float16),
                ct.TensorType(name="new_conv_states", dtype=np.float16),
                ct.TensorType(name="new_ssm_states", dtype=np.float16),
            ],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
        )
        print("OK")

        if not args.no_quantize:
            print("  LUT6...", end=" ", flush=True)
            mlmodel = palettize_weights(mlmodel, LUT6_CONFIG)
            print("OK")

        out_path = os.path.join(args.output, f"hybrid_chunk_{bi}_mid.mlpackage")
        mlmodel.save(out_path)
        size_mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(out_path) for f in fns) / 1e6
        print(f"  Saved: {out_path} ({size_mb:.1f} MB)")

        # Benchmark
        m = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        inp = {
            "attn_output": np.random.randn(1, attn_dim).astype(np.float16),
            "prev_residual": np.random.randn(1, hidden).astype(np.float16),
            "conv_states": np.zeros((n_dn_per_block, lin_qkv_dim, conv_kernel), dtype=np.float16),
            "ssm_states": np.zeros((n_dn_per_block, lin_num_val_heads, lin_key_dim, lin_val_dim), dtype=np.float16),
        }
        for _ in range(5):
            m.predict(inp)
        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            m.predict(inp)
        avg_ms = (time.perf_counter() - t0) / N_BENCH * 1000
        print(f"  Benchmark: {avg_ms:.2f} ms")

    # ==================== Final chunk ====================
    last_fa = super_blocks[-1]["fa"]
    print(f"\n=== Final chunk: post-attn L{last_fa} + final_norm ===")

    post_attn_final = load_post_attn(handles, weight_map, prefix, last_fa, tc)
    eps = tc.get("rms_norm_eps", 1e-6)
    final_norm = RMSNorm(hidden, eps)
    with torch.no_grad():
        final_norm.weight.copy_(get_tensor(handles, weight_map, f"{prefix}norm.weight"))

    final = FinalChunk(post_attn_final, final_norm)
    final.eval()

    print("  Tracing...", end=" ", flush=True)
    with torch.no_grad():
        traced = torch.jit.trace(final, (attn_out_ex, res_ex))
    print("OK")

    print("  Converting...", end=" ", flush=True)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="attn_output", shape=(1, attn_dim), dtype=np.float16),
            ct.TensorType(name="prev_residual", shape=(1, hidden), dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="hidden_states", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    print("OK")

    if not args.no_quantize:
        print("  LUT6...", end=" ", flush=True)
        mlmodel = palettize_weights(mlmodel, LUT6_CONFIG)
        print("OK")

    out_path = os.path.join(args.output, "hybrid_chunk_final.mlpackage")
    mlmodel.save(out_path)
    size_mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(out_path) for f in fns) / 1e6
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")

    # Benchmark
    m = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    inp = {
        "attn_output": np.random.randn(1, attn_dim).astype(np.float16),
        "prev_residual": np.random.randn(1, hidden).astype(np.float16),
    }
    for _ in range(5):
        m.predict(inp)
    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        m.predict(inp)
    avg_ms = (time.perf_counter() - t0) / N_BENCH * 1000
    print(f"  Benchmark: {avg_ms:.2f} ms")

    # ==================== LM head with in-model argmax ====================
    if args.lm_head:
        vocab_size = tc["vocab_size"]
        topk = args.topk
        print(f"\n=== LM Head (in-model top-{topk} argmax): {hidden} → {vocab_size} ===")

        tie = tc.get("tie_word_embeddings", False)
        lm_key = f"{prefix}embed_tokens.weight" if tie else f"{prefix}lm_head.weight"
        lm_w = get_tensor(handles, weight_map, lm_key)

        lm_head = LMHeadArgmax(hidden, vocab_size, topk)
        with torch.no_grad():
            lm_head.proj.weight.copy_(lm_w)
        lm_head.eval()

        print("  Tracing...", end=" ", flush=True)
        with torch.no_grad():
            traced = torch.jit.trace(lm_head, (torch.randn(1, hidden),))
        print("OK")

        print("  Converting...", end=" ", flush=True)
        lm_mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input", shape=(1, hidden), dtype=np.float16)],
            outputs=[
                ct.TensorType(name="top_logits", dtype=np.float16),
                ct.TensorType(name="top_indices", dtype=np.float32),
            ],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
        )
        print("OK")

        if not args.no_quantize:
            print("  LUT6...", end=" ", flush=True)
            lm_mlmodel = palettize_weights(lm_mlmodel, LUT6_CONFIG)
            print("OK")

        lm_path = os.path.join(args.output, "lm_head_topk.mlpackage")
        lm_mlmodel.save(lm_path)
        size_mb = sum(os.path.getsize(os.path.join(dp, f))
                      for dp, _, fns in os.walk(lm_path) for f in fns) / 1e6
        print(f"  Saved: {lm_path} ({size_mb:.1f} MB)")

        # Benchmark
        m = ct.models.MLModel(lm_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        inp = {"input": np.random.randn(1, hidden).astype(np.float16)}
        for _ in range(5):
            m.predict(inp)
        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            m.predict(inp)
        avg_ms = (time.perf_counter() - t0) / N_BENCH * 1000
        print(f"  Benchmark: {avg_ms:.2f} ms (was ~3.7ms returning full logits)")

    print(f"\n=== Summary ===")
    print(f"Total chunks: 1 first + {n_blocks-1} mid + 1 final = {n_blocks + 1}")
    print(f"CPU attention calls: {n_blocks}")
    print(f"Cross-device boundaries: {n_blocks}")
    print(f"\nPipeline: embed → chunk_0 → cpu_attn → chunk_1 → cpu_attn → ... → chunk_final → lm_head")


if __name__ == "__main__":
    main()
