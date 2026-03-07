#!/usr/bin/env python3
"""Export all Qwen3.5 matmul operations as CoreML models with LUT6 quantization.

Default (norm-fused) mode creates 3 CoreML models per layer:
    - layer_N_norm_first_proj.mlpackage  (RMSNorm + fused QKV/QKVZ + in_proj_a/b)
    - layer_N_o_proj.mlpackage           (output projection)
    - layer_N_norm_fused_ffn.mlpackage   (RMSNorm + gate+up -> SiLU -> down)

Legacy mode (--legacy) creates 3 CoreML models per layer:
    - layer_N_first_proj.mlpackage  (fused QKV/QKVZ projection)
    - layer_N_o_proj.mlpackage      (output projection)
    - layer_N_fused_ffn.mlpackage   (gate+up -> SiLU -> down)

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
# Larger chunks = fewer CoreML calls = lower per-token overhead.
# 262144 covers 248K vocab in a single chunk for 0.8B model.
LM_HEAD_CHUNK_MAX = 262144

# Batch sizes for prefill batching (EnumeratedShapes)
BATCH_SIZES = [1, 2, 4, 8]


# ============ PyTorch Modules ============


class NormConcatMatmul(nn.Module):
    """RMSNorm + concatenated matmul (fused input_layernorm + first_proj).

    For DeltaNet layers, the projection includes in_proj_a/b as extra output rows.
    Input is raw x (not pre-normed).
    """
    def __init__(self, hidden_size, out_dim, eps=1e-6):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.proj = nn.Linear(hidden_size, out_dim, bias=False)
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps) * self.norm_weight
        return self.proj(x_norm)


class NormFusedFFN(nn.Module):
    """RMSNorm + gate+up -> SiLU -> down (fused post_attention_layernorm + FFN).

    Input is raw x (post-residual, not pre-normed).
    """
    def __init__(self, hidden_size, intermediate_size, eps=1e-6):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps) * self.norm_weight
        return self.down_proj(nn.functional.silu(self.gate_proj(x_norm)) * self.up_proj(x_norm))


class FusedFFN(nn.Module):
    """Gate + Up -> SiLU -> Down FFN (legacy, no norm)."""
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


# ============ Helpers ============


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
        weight_map = {}
        for name, handle in handles.items():
            for key in handle.keys():
                weight_map[key] = name
        return handles, weight_map


def get_tensor(handles, weight_map, key):
    """Load a tensor by key from the correct shard."""
    shard = weight_map[key]
    return handles[shard].get_tensor(key).to(torch.float32)


def convert_and_save(module, input_shape, output_path, name, batch_sizes=None):
    """Convert a PyTorch module to CoreML with LUT6 and save.

    If batch_sizes is provided (e.g. [1,2,4,8]), exports with EnumeratedShapes
    so the model accepts multiple batch dimensions for prefill batching.
    """
    module.eval()
    example = torch.randn(*input_shape)
    traced = torch.jit.trace(module, example)

    if batch_sizes and len(batch_sizes) > 1:
        in_dim = input_shape[-1]
        shapes = [ct.Shape((b, in_dim)) for b in batch_sizes]
        input_type = ct.TensorType(
            name="input",
            shape=ct.EnumeratedShapes(shapes=shapes),
            dtype=np.float16,
        )
    else:
        input_type = ct.TensorType(name="input", shape=input_shape, dtype=np.float16)

    mlmodel = ct.convert(
        traced,
        inputs=[input_type],
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


# ============ Layer Exporters ============


def export_layer_norm_fused(handles, weight_map, prefix, layer_idx, layer_type, config, output_dir,
                            batch_sizes=None):
    """Export norm-fused models for a single layer.

    Exports:
      - norm_first_proj: RMSNorm + QKV projection (+ in_proj_a/b for DeltaNet)
      - o_proj: output projection (unchanged)
      - norm_fused_ffn: RMSNorm + gate+up -> SiLU -> down
    """
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    eps = config.get("rms_norm_eps", 1e-6)

    print(f"  Layer {layer_idx} ({layer_type})...")

    # --- norm_first_proj: RMSNorm + first_proj ---
    # +1.0: model stores residual norm weights (w-1); actual weight is w_stored + 1
    norm_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.input_layernorm.weight") + 1.0

    if layer_type == "linear_attention":
        # DeltaNet: fuse QKV + Z + in_proj_a + in_proj_b
        qkv_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.linear_attn.in_proj_qkv.weight")
        z_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.linear_attn.in_proj_z.weight")
        a_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.linear_attn.in_proj_a.weight")
        b_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.linear_attn.in_proj_b.weight")
        fused_w = torch.cat([qkv_w, z_w, a_w, b_w], dim=0)
        out_dim = fused_w.shape[0]
    else:
        # FullAttention: fuse Q + K + V
        q_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.self_attn.q_proj.weight")
        k_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.self_attn.k_proj.weight")
        v_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.self_attn.v_proj.weight")
        fused_w = torch.cat([q_w, k_w, v_w], dim=0)
        out_dim = fused_w.shape[0]

    module = NormConcatMatmul(hidden, out_dim, eps)
    module.norm_weight.data = norm_w
    module.proj.weight.data = fused_w
    convert_and_save(
        module, (1, hidden),
        os.path.join(output_dir, f"layer_{layer_idx}_norm_first_proj.mlpackage"),
        f"norm_first_proj ({hidden}->{out_dim})",
        batch_sizes=batch_sizes,
    )

    # --- o_proj (unchanged) ---
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
        f"o_proj ({o_in}->{o_out})",
        batch_sizes=batch_sizes,
    )

    # --- norm_fused_ffn: RMSNorm + FFN ---
    # +1.0: residual norm correction (same as input_layernorm)
    post_norm_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.post_attention_layernorm.weight") + 1.0
    gate_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.mlp.gate_proj.weight")
    up_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.mlp.up_proj.weight")
    down_w = get_tensor(handles, weight_map, f"{prefix}layers.{layer_idx}.mlp.down_proj.weight")

    module = NormFusedFFN(hidden, intermediate, eps)
    module.norm_weight.data = post_norm_w
    module.gate_proj.weight.data = gate_w
    module.up_proj.weight.data = up_w
    module.down_proj.weight.data = down_w
    convert_and_save(
        module, (1, hidden),
        os.path.join(output_dir, f"layer_{layer_idx}_norm_fused_ffn.mlpackage"),
        f"norm_fused_ffn ({hidden}->{intermediate}->{hidden})",
        batch_sizes=batch_sizes,
    )


def export_layer(handles, weight_map, prefix, layer_idx, layer_type, config, output_dir):
    """Export legacy (no norm fusion) models for a single layer."""
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]

    print(f"  Layer {layer_idx} ({layer_type})...")

    # --- first_proj ---
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
        f"o_proj ({o_in}->{o_out})",
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
        f"fused_ffn ({hidden}->{intermediate}->{hidden})",
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
            f"lm_head chunk {c}/{n_chunks} ({hidden}->{rows})",
        )


# ============ Main ============


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3.5 model as CoreML with LUT6")
    parser.add_argument("--model", required=True, help="Path to Qwen3.5 model directory")
    parser.add_argument("--output", required=True, help="Output directory for CoreML models")
    parser.add_argument("--skip-lm-head", action="store_true", help="Skip LM head export")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--norm-fused", action="store_true", default=True,
                            help="Fuse RMSNorm into projections (default)")
    mode_group.add_argument("--legacy", action="store_true",
                            help="Legacy mode: separate norm + projection")
    parser.add_argument("--batch-prefill", action="store_true",
                        help="Export with EnumeratedShapes for batched prefill "
                             "(76%% faster prefill but 15%% slower generation)")
    args = parser.parse_args()

    norm_fused = not args.legacy

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
    print(f"  Mode: {'norm-fused' if norm_fused else 'legacy'}")
    print()

    # Open safetensors
    handles, weight_map = open_safetensors(args.model)
    prefix = detect_weight_prefix(list(handles.values()))
    print(f"Weight prefix: '{prefix}'")
    print()

    # Export all layers
    batch_sizes = BATCH_SIZES if args.batch_prefill else None
    if batch_sizes:
        print(f"  Batch prefill: EnumeratedShapes {batch_sizes}")
    t_start = time.time()
    for L in range(num_layers):
        if norm_fused:
            export_layer_norm_fused(handles, weight_map, prefix, L, layer_types[L], config, args.output,
                                   batch_sizes=batch_sizes)
        else:
            export_layer(handles, weight_map, prefix, L, layer_types[L], config, args.output)

    # Export LM head
    if not args.skip_lm_head:
        export_lm_head(handles, weight_map, prefix, config, args.output, tie_embeddings)

    elapsed = time.time() - t_start
    n_models = num_layers * 3 + (0 if args.skip_lm_head else
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
        "norm_fused": norm_fused,
        "lm_head_chunks": (config["vocab_size"] + LM_HEAD_CHUNK_MAX - 1) // LM_HEAD_CHUNK_MAX
                          if not args.skip_lm_head else 0,
        "batch_sizes": BATCH_SIZES if args.batch_prefill else [1],
    }
    with open(os.path.join(args.output, "coreml_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: coreml_meta.json")


if __name__ == "__main__":
    main()
