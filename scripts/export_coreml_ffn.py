#!/usr/bin/env python3
"""Export a single FFN layer from Qwen3.5 as a CoreML model with LUT6 quantization.

Usage:
    python scripts/export_coreml_ffn.py --model models/Qwen3.5-0.8B --output models/coreml_bench

Exports:
    - fused_ffn_layer0.mlpackage  (gate+up → SiLU → down, LUT6)
    - fused_ffn_layer0_fp16.mlpackage  (same, no quantization)
    - o_proj_layer0.mlpackage  (simple matmul, LUT6)
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


class FusedFFN(nn.Module):
    """Gate + Up → SiLU → Down FFN (matches Qwen3.5 MLP)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class SimpleMatmul(nn.Module):
    """Single linear projection (for o_proj benchmark)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


def detect_weight_prefix(st_file):
    """Detect the weight prefix used in safetensors (same logic as C++)."""
    prefixes = ["model.language_model.", "model.", ""]
    for pfx in prefixes:
        key = f"{pfx}embed_tokens.weight"
        try:
            st_file.get_tensor(key)
            return pfx
        except Exception:
            continue
    raise ValueError("Cannot detect weight prefix")


def load_ffn_weights(model_dir: str, layer: int = 0):
    """Load FFN weights from safetensors."""
    # Find safetensors files
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        # Find which shard has our weights
        shards = set()
        for key, shard in index["weight_map"].items():
            if f"layers.{layer}.mlp" in key or f"layers.{layer}.linear_attn.out_proj" in key:
                shards.add(shard)
        shard_files = [os.path.join(model_dir, s) for s in shards]
    else:
        shard_files = [
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if f.endswith(".safetensors")
        ]

    # Open all shards and detect prefix
    st = safe_open(shard_files[0], framework="torch")
    prefix = detect_weight_prefix(st)
    print(f"Weight prefix: '{prefix}'")

    # Load weights
    gate_key = f"{prefix}layers.{layer}.mlp.gate_proj.weight"
    up_key = f"{prefix}layers.{layer}.mlp.up_proj.weight"
    down_key = f"{prefix}layers.{layer}.mlp.down_proj.weight"
    o_proj_key = f"{prefix}layers.{layer}.linear_attn.out_proj.weight"

    weights = {}
    for sf_path in shard_files:
        sf = safe_open(sf_path, framework="torch")
        for key in [gate_key, up_key, down_key, o_proj_key]:
            try:
                weights[key] = sf.get_tensor(key).to(torch.float32)
            except Exception:
                pass

    return weights, prefix, layer


def export_ffn(weights, prefix, layer, hidden_size, intermediate_size, output_dir, quantize=True):
    """Export FFN as CoreML model."""
    gate_key = f"{prefix}layers.{layer}.mlp.gate_proj.weight"
    up_key = f"{prefix}layers.{layer}.mlp.up_proj.weight"
    down_key = f"{prefix}layers.{layer}.mlp.down_proj.weight"

    # Build PyTorch module
    ffn = FusedFFN(hidden_size, intermediate_size)
    ffn.gate_proj.weight.data = weights[gate_key]
    ffn.up_proj.weight.data = weights[up_key]
    ffn.down_proj.weight.data = weights[down_key]
    ffn.eval()

    # Trace
    example_input = torch.randn(1, hidden_size)
    traced = torch.jit.trace(ffn, example_input)

    # Convert to CoreML
    print(f"Converting FFN to CoreML (quantize={quantize})...")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, hidden_size))],
        outputs=[ct.TensorType(name="output")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )
    print(f"  Conversion took {time.time() - t0:.1f}s")

    if quantize:
        print("  Applying LUT6 palettization...")
        t0 = time.time()
        config = OptimizationConfig(global_config=OpPalettizerConfig(nbits=6))
        mlmodel = palettize_weights(mlmodel, config)
        print(f"  Palettization took {time.time() - t0:.1f}s")
        suffix = ""
    else:
        suffix = "_fp16"

    # Save
    out_path = os.path.join(output_dir, f"fused_ffn_layer{layer}{suffix}.mlpackage")
    mlmodel.save(out_path)
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fn in os.walk(out_path)
        for f in fn
    ) / 1024 / 1024
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")
    return out_path


def export_o_proj(weights, prefix, layer, output_dir):
    """Export o_proj as simple matmul CoreML model with LUT6."""
    o_proj_key = f"{prefix}layers.{layer}.linear_attn.out_proj.weight"
    w = weights[o_proj_key]
    out_dim, in_dim = w.shape

    proj = SimpleMatmul(in_dim, out_dim)
    proj.proj.weight.data = w
    proj.eval()

    example_input = torch.randn(1, in_dim)
    traced = torch.jit.trace(proj, example_input)

    print(f"Converting o_proj ({in_dim}→{out_dim}) to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, in_dim))],
        outputs=[ct.TensorType(name="output")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )

    config = OptimizationConfig(global_config=OpPalettizerConfig(nbits=6))
    mlmodel = palettize_weights(mlmodel, config)

    out_path = os.path.join(output_dir, f"o_proj_layer{layer}.mlpackage")
    mlmodel.save(out_path)
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fn in os.walk(out_path)
        for f in fn
    ) / 1024 / 1024
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3.5 FFN layer as CoreML model")
    parser.add_argument("--model", required=True, help="Path to Qwen3.5 model directory")
    parser.add_argument("--output", required=True, help="Output directory for CoreML models")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to export")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Read config
    config_path = os.path.join(args.model, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    text_config = config.get("text_config", config)
    hidden_size = text_config["hidden_size"]
    intermediate_size = text_config["intermediate_size"]
    print(f"Model: hidden_size={hidden_size}, intermediate_size={intermediate_size}")

    # Load weights
    print(f"Loading layer {args.layer} weights...")
    weights, prefix, layer = load_ffn_weights(args.model, args.layer)
    print(f"  Loaded {len(weights)} weight tensors")

    # Export FFN with LUT6
    export_ffn(weights, prefix, layer, hidden_size, intermediate_size, args.output, quantize=True)

    # Export FFN without quantization (FP16 baseline)
    export_ffn(weights, prefix, layer, hidden_size, intermediate_size, args.output, quantize=False)

    # Export o_proj with LUT6
    export_o_proj(weights, prefix, layer, args.output)

    print("\nDone! Run bench_coreml.py to benchmark.")


if __name__ == "__main__":
    main()
