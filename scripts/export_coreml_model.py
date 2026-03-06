#!/usr/bin/env python3
"""Export all Qwen3.5 matmul operations as CoreML models with LUT6 quantization.

Default mode creates 3 CoreML models per layer:
    - layer_N_first_proj.mlpackage  (fused QKV/QKVZ projection)
    - layer_N_o_proj.mlpackage      (output projection)
    - layer_N_fused_ffn.mlpackage   (gate+up → SiLU → down)

Experimental --fused mode creates 2 CoreML models per layer:
    - layer_N_first_proj.mlpackage  (fused QKV/QKVZ projection)
    - layer_N_post_attn.mlpackage   (o_proj + residual + norm + FFN + residual)
    NOTE: Fused mode has FP16 precision issues on ANE (garbled output with small
    hidden values). Not recommended for production use.

Both modes also export:
    - lm_head_chunk_C.mlpackage     (vocabulary projection chunks)

Usage:
    python scripts/export_coreml_model.py \\
        --model models/Qwen3.5-0.8B \\
        --output models/Qwen3.5-0.8B-coreml
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from coremltools.optimize.coreml import (
    palettize_weights,
    OpPalettizerConfig,
    OptimizationConfig,
)
from safetensors import safe_open


# LUT6 quantization config (applied to all models)
LUT6_CONFIG = OptimizationConfig(global_config=OpPalettizerConfig(nbits=6))

# Max chunk size for LM head (matches C++ LM_HEAD_ANE_CHUNK_MAX)
LM_HEAD_CHUNK_MAX = 16384


class FusedFFN(nn.Module):
    """Gate + Up → SiLU → Down FFN."""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class ConcatMatmul(nn.Module):
    """Concatenated matmul: produces [W1 @ x ; W2 @ x ; ...] as single output."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


class SimpleMatmul(nn.Module):
    """Single linear projection."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


class PostAttnFused(nn.Module):
    """Fused o_proj + residual + RMSNorm + FFN + residual.

    Takes two inputs: attention output and residual state.
    Replaces separate o_proj and fused_ffn CoreML calls with a single call.
    """
    def __init__(self, attn_dim, hidden_size, intermediate_size, eps=1e-6):
        super().__init__()
        self.o_proj = nn.Linear(attn_dim, hidden_size, bias=False)
        self.norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.eps = eps

    def forward(self, attn_output, x_residual):
        o = self.o_proj(attn_output)
        h = x_residual + o
        # Scale up for FP16 precision in RMSNorm: near-zero embedding values
        # (~0.01) cause variance ≈ 0.0001 which has poor FP16 mantissa precision.
        # Scaling cancels mathematically: norm(s*x) = s*x / (s * ||x||) = x / ||x||
        SCALE = 128.0
        h_scaled = h * SCALE
        variance = h_scaled.pow(2).mean(-1, keepdim=True)
        h_norm = h_scaled * torch.rsqrt(variance + self.eps * SCALE * SCALE) * self.norm_weight
        ffn_out = self.down_proj(
            nn.functional.silu(self.gate_proj(h_norm)) * self.up_proj(h_norm)
        )
        return h + ffn_out


def detect_weight_prefix(st_files):
    """Detect weight prefix from safetensors files."""
    prefixes = ["model.language_model.", "model.", ""]
    for st in st_files:
        for pfx in prefixes:
            try:
                st.get_tensor(f"{pfx}embed_tokens.weight")
                return pfx
            except Exception:
                continue
    raise ValueError("Cannot detect weight prefix")


def open_safetensors(model_dir):
    """Open all safetensors files and return (list_of_handles, weight_map)."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_names = sorted(set(index["weight_map"].values()))
        handles = {
            name: safe_open(os.path.join(model_dir, name), framework="torch")
            for name in shard_names
        }
        weight_map = index["weight_map"]
        return handles, weight_map
    else:
        files = sorted(
            f for f in os.listdir(model_dir) if f.endswith(".safetensors")
        )
        handles = {
            f: safe_open(os.path.join(model_dir, f), framework="torch")
            for f in files
        }
        # Build weight map from available tensors
        weight_map = {}
        for name, handle in handles.items():
            for key in handle.keys():
                weight_map[key] = name
        return handles, weight_map


def get_tensor(handles, weight_map, key):
    """Load a tensor by key from the correct shard."""
    shard = weight_map[key]
    return handles[shard].get_tensor(key).to(torch.float32)


def convert_and_save(module, input_shape, output_path, name):
    """Convert a PyTorch module to CoreML with LUT6 and save."""
    module.eval()
    example = torch.randn(*input_shape)
    traced = torch.jit.trace(module, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=input_shape)],
        outputs=[ct.TensorType(name="output")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )

    mlmodel = palettize_weights(mlmodel, LUT6_CONFIG)
    mlmodel.save(output_path)

    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fn in os.walk(output_path)
        for f in fn
    ) / 1024 / 1024
    print(f"    {name}: {size_mb:.1f} MB")


def convert_and_save_2input(module, shape1, shape2, output_path, name):
    """Convert a 2-input PyTorch module to CoreML with LUT6 and save."""
    module.eval()
    ex1 = torch.randn(*shape1)
    ex2 = torch.randn(*shape2)
    traced = torch.jit.trace(module, (ex1, ex2))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="attn_output", shape=shape1),
            ct.TensorType(name="x_residual", shape=shape2),
        ],
        outputs=[ct.TensorType(name="output")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )

    mlmodel = palettize_weights(mlmodel, LUT6_CONFIG)
    mlmodel.save(output_path)

    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fn in os.walk(output_path)
        for f in fn
    ) / 1024 / 1024
    print(f"    {name}: {size_mb:.1f} MB")


def export_layer(handles, weight_map, prefix, layer_idx, layer_type, config, output_dir):
    """Export all matmul ops for a single layer."""
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]

    print(f"  Layer {layer_idx} ({layer_type})...")

    # --- first_proj ---
    if layer_type == "linear_attention":
        # DeltaNet: fused QKV + Z projection
        qkv_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.linear_attn.in_proj_qkv.weight")
        z_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.linear_attn.in_proj_z.weight")
        # Concatenate: output = [QKV ; Z]
        fused_w = torch.cat([qkv_w, z_w], dim=0)
        out_dim = fused_w.shape[0]

        module = ConcatMatmul(hidden, out_dim)
        module.proj.weight.data = fused_w
        convert_and_save(
            module, (1, hidden),
            os.path.join(output_dir, f"layer_{layer_idx}_first_proj.mlpackage"),
            f"first_proj ({hidden}→{out_dim})",
        )
    else:
        # FullAttention: fused Q + K + V projection
        q_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.self_attn.q_proj.weight")
        k_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.self_attn.k_proj.weight")
        v_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.self_attn.v_proj.weight")
        fused_w = torch.cat([q_w, k_w, v_w], dim=0)
        out_dim = fused_w.shape[0]

        module = ConcatMatmul(hidden, out_dim)
        module.proj.weight.data = fused_w
        convert_and_save(
            module, (1, hidden),
            os.path.join(output_dir, f"layer_{layer_idx}_first_proj.mlpackage"),
            f"first_proj ({hidden}→{out_dim})",
        )

    # --- o_proj ---
    if layer_type == "linear_attention":
        o_key = f"{prefix}layers.{layer_idx}.linear_attn.out_proj.weight"
    else:
        o_key = f"{prefix}layers.{layer_idx}.self_attn.o_proj.weight"

    o_w = get_tensor(handles, weight_map, o_key)
    o_out, o_in = o_w.shape

    module = SimpleMatmul(o_in, o_out)
    module.proj.weight.data = o_w
    convert_and_save(
        module, (1, o_in),
        os.path.join(output_dir, f"layer_{layer_idx}_o_proj.mlpackage"),
        f"o_proj ({o_in}→{o_out})",
    )

    # --- fused_ffn ---
    gate_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.mlp.gate_proj.weight")
    up_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.mlp.up_proj.weight")
    down_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.mlp.down_proj.weight")

    module = FusedFFN(hidden, intermediate)
    module.gate_proj.weight.data = gate_w
    module.up_proj.weight.data = up_w
    module.down_proj.weight.data = down_w
    convert_and_save(
        module, (1, hidden),
        os.path.join(output_dir, f"layer_{layer_idx}_fused_ffn.mlpackage"),
        f"fused_ffn ({hidden}→{intermediate}→{hidden})",
    )


def export_layer_fused(handles, weight_map, prefix, layer_idx, layer_type, config, output_dir):
    """Export first_proj + fused post_attn for a single layer."""
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    eps = config.get("rms_norm_eps", 1e-6)

    print(f"  Layer {layer_idx} ({layer_type})...")

    # --- first_proj (same as non-fused) ---
    if layer_type == "linear_attention":
        qkv_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.linear_attn.in_proj_qkv.weight")
        z_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.linear_attn.in_proj_z.weight")
        fused_w = torch.cat([qkv_w, z_w], dim=0)
        out_dim = fused_w.shape[0]
        module = ConcatMatmul(hidden, out_dim)
        module.proj.weight.data = fused_w
        convert_and_save(
            module, (1, hidden),
            os.path.join(output_dir, f"layer_{layer_idx}_first_proj.mlpackage"),
            f"first_proj ({hidden}->{out_dim})",
        )
    else:
        q_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.self_attn.q_proj.weight")
        k_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.self_attn.k_proj.weight")
        v_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.self_attn.v_proj.weight")
        fused_w = torch.cat([q_w, k_w, v_w], dim=0)
        out_dim = fused_w.shape[0]
        module = ConcatMatmul(hidden, out_dim)
        module.proj.weight.data = fused_w
        convert_and_save(
            module, (1, hidden),
            os.path.join(output_dir, f"layer_{layer_idx}_first_proj.mlpackage"),
            f"first_proj ({hidden}->{out_dim})",
        )

    # --- post_attn (fused o_proj + residual + norm + FFN + residual) ---
    if layer_type == "linear_attention":
        o_key = f"{prefix}layers.{layer_idx}.linear_attn.out_proj.weight"
    else:
        o_key = f"{prefix}layers.{layer_idx}.self_attn.o_proj.weight"

    o_w = get_tensor(handles, weight_map, o_key)
    attn_dim = o_w.shape[1]

    norm_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.post_attention_layernorm.weight")
    gate_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.mlp.gate_proj.weight")
    up_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.mlp.up_proj.weight")
    down_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.mlp.down_proj.weight")

    module = PostAttnFused(attn_dim, hidden, intermediate, eps)
    module.o_proj.weight.data = o_w
    module.norm_weight.data = norm_w
    module.gate_proj.weight.data = gate_w
    module.up_proj.weight.data = up_w
    module.down_proj.weight.data = down_w
    convert_and_save_2input(
        module, (1, attn_dim), (1, hidden),
        os.path.join(output_dir, f"layer_{layer_idx}_post_attn.mlpackage"),
        f"post_attn ({attn_dim}+{hidden}->{hidden})",
    )


def export_lm_head(handles, weight_map, prefix, config, output_dir, tie_embeddings):
    """Export LM head as chunked matmul models."""
    vocab_size = config["vocab_size"]
    hidden = config["hidden_size"]

    if tie_embeddings:
        lm_head_key = f"{prefix}embed_tokens.weight"
    else:
        lm_head_key = f"{prefix}lm_head.weight"

    lm_w = get_tensor(handles, weight_map, lm_head_key)
    print(f"  LM head: {lm_w.shape} (vocab={vocab_size})")

    chunk_size = LM_HEAD_CHUNK_MAX
    n_chunks = (vocab_size + chunk_size - 1) // chunk_size

    for c in range(n_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, vocab_size)
        rows = end - start

        module = SimpleMatmul(hidden, rows)
        module.proj.weight.data = lm_w[start:end]
        convert_and_save(
            module, (1, hidden),
            os.path.join(output_dir, f"lm_head_chunk_{c}.mlpackage"),
            f"lm_head chunk {c}/{n_chunks} ({hidden}→{rows})",
        )


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3.5 model as CoreML with LUT6")
    parser.add_argument("--model", required=True, help="Path to Qwen3.5 model directory")
    parser.add_argument("--output", required=True, help="Output directory for CoreML models")
    parser.add_argument("--skip-lm-head", action="store_true", help="Skip LM head export")
    parser.add_argument("--fused", action="store_true",
                        help="Experimental: fuse o_proj+norm+FFN into single 2-input model (has FP16 precision issues)")
    args = parser.parse_args()
    fused = args.fused

    os.makedirs(args.output, exist_ok=True)

    # Read config
    with open(os.path.join(args.model, "config.json")) as f:
        full_config = json.load(f)
    config = full_config.get("text_config", full_config)
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    num_layers = config["num_hidden_layers"]
    layer_types = config["layer_types"]
    tie_embeddings = full_config.get("tie_word_embeddings", True)

    print(f"Model: {args.model}")
    print(f"  hidden_size={hidden}, intermediate_size={intermediate}")
    print(f"  {num_layers} layers, tie_embeddings={tie_embeddings}")
    print(f"  Layer types: {sum(1 for t in layer_types if t == 'linear_attention')} DeltaNet + "
          f"{sum(1 for t in layer_types if t == 'full_attention')} FullAttention")
    print(f"  Mode: {'fused (2 calls/layer)' if fused else 'legacy (3 calls/layer)'}")
    print()

    # Open safetensors
    handles, weight_map = open_safetensors(args.model)
    prefix = detect_weight_prefix(list(handles.values()))
    print(f"Weight prefix: '{prefix}'")
    print()

    # Export all layers
    t_start = time.time()
    for L in range(num_layers):
        if fused:
            export_layer_fused(handles, weight_map, prefix, L, layer_types[L], config, args.output)
        else:
            export_layer(handles, weight_map, prefix, L, layer_types[L], config, args.output)

    # Export LM head
    if not args.skip_lm_head:
        export_lm_head(handles, weight_map, prefix, config, args.output, tie_embeddings)

    elapsed = time.time() - t_start
    models_per_layer = 2 if fused else 3
    n_models = num_layers * models_per_layer + (0 if args.skip_lm_head else
               (config["vocab_size"] + LM_HEAD_CHUNK_MAX - 1) // LM_HEAD_CHUNK_MAX)
    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fn in os.walk(args.output)
        for f in fn
    ) / 1024 / 1024

    print(f"\nDone! {n_models} CoreML models exported in {elapsed:.1f}s")
    print(f"Total size: {total_size:.1f} MB")
    print(f"Output: {args.output}")

    # Save metadata for C++ loader
    meta = {
        "model_dir": os.path.basename(args.model),
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_layers": num_layers,
        "layer_types": layer_types,
        "quantization": "lut6",
        "fused": fused,
        "lm_head_chunks": (config["vocab_size"] + LM_HEAD_CHUNK_MAX - 1) // LM_HEAD_CHUNK_MAX
                          if not args.skip_lm_head else 0,
    }
    with open(os.path.join(args.output, "coreml_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: coreml_meta.json")


if __name__ == "__main__":
    main()
