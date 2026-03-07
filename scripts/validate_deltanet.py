#!/usr/bin/env python3
"""Validate fused DeltaNetLayer against transformers gold reference.

Runs both implementations step-by-step on the same input tokens,
comparing per-layer per-token outputs to pinpoint where divergence starts.

Usage:
    cd /Users/hb/projects/ANE-LM
    python scripts/validate_deltanet.py --model models/Qwen3.5-4B --tokens 20
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


# ============ Fused DeltaNetLayer (from export_hybrid_chunks.py) ============

class RMSNorm(nn.Module):
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


def fused_l2norm(x, dim=-1, eps=1e-6):
    """Match the transformers l2norm exactly: x * rsqrt(sum(x*x) + eps)."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


class FusedDeltaNetLayer(nn.Module):
    """Fused DeltaNet layer matching export_hybrid_chunks.py but with l2norm fix."""
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

        # Conv1d update
        new_conv_state = torch.cat([conv_state[:, 1:], mixed_qkv.squeeze(0).unsqueeze(1)], dim=1)
        conv_out = (new_conv_state * self.conv_weight).sum(dim=1)
        conv_out = F.silu(conv_out)

        # Split Q, K, V
        key_total = self.lin_num_key_heads * self.lin_key_dim
        q = conv_out[:key_total].view(self.lin_num_key_heads, self.lin_key_dim)
        k = conv_out[key_total:2*key_total].view(self.lin_num_key_heads, self.lin_key_dim)
        v = conv_out[2*key_total:].view(self.lin_num_val_heads, self.lin_val_dim)

        # Repeat interleave
        q = q.repeat_interleave(self.head_ratio, dim=0)
        k = k.repeat_interleave(self.head_ratio, dim=0)

        # Use matching l2norm (not F.normalize)
        q = fused_l2norm(q, dim=-1, eps=1e-6)
        k = fused_l2norm(k, dim=-1, eps=1e-6)

        # Scale
        scale = 1.0 / (self.lin_key_dim ** 0.5)
        q = q * scale

        # Gating
        beta = torch.sigmoid(b_raw).squeeze(0)
        g = (-torch.exp(self.A_log.float()) * F.softplus(a_raw.float().squeeze(0) + self.dt_bias))
        decay = torch.exp(g)

        # SSM step (matches torch_recurrent_gated_delta_rule)
        decayed_state = decay.view(-1, 1, 1) * ssm_state
        Sk = torch.einsum('hkv,hk->hv', decayed_state, k)
        delta = (v - Sk) * beta.view(-1, 1)
        outer = k.unsqueeze(2) * delta.unsqueeze(1)
        new_ssm_state = decayed_state + outer
        y = torch.einsum('hkv,hk->hv', new_ssm_state, q)

        # Gated RMSNorm (match reference: norm in fp32, then weight, then silu gate in fp32)
        y_fp32 = y.float()
        var = y_fp32.pow(2).mean(dim=-1, keepdim=True)
        y_norm = y_fp32 * torch.rsqrt(var + 1e-6)
        y_norm = self.norm_weight.unsqueeze(0) * y_norm
        z_reshaped = z.view(self.lin_num_val_heads, self.lin_val_dim)
        y_normed = (y_norm * F.silu(z_reshaped.float())).reshape(-1)

        attn_out = self.out_proj(y_normed.unsqueeze(0))
        x = residual + attn_out
        residual2 = x
        x = residual2 + self.ffn(self.post_attention_layernorm(x))
        return x, new_conv_state, new_ssm_state


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


def load_fused_deltanet(handles, weight_map, prefix, layer_idx, config):
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

    layer = FusedDeltaNetLayer(
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


# ============ Reference Model ============

def load_reference_model(model_dir):
    """Load model with transformers for gold reference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[Ref] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    print("[Ref] Loading model (float32)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


# ============ Step-by-step Reference Decode ============

def reference_step_by_step(model, input_ids, layer_indices, num_tokens):
    """Run reference model token-by-token, capturing per-layer hidden states.

    Returns dict: {layer_idx: [hidden_after_layer_for_each_token]}
    Also returns final logits for each decode step.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

    text_model = model.model if hasattr(model, 'model') else model
    # Navigate to the text model layers
    if hasattr(text_model, 'language_model'):
        text_model = text_model.language_model
    if hasattr(text_model, 'model'):
        inner = text_model.model
    else:
        inner = text_model

    config = inner.config if hasattr(inner, 'config') else model.config
    text_config = config.text_config if hasattr(config, 'text_config') else config

    layers = inner.layers
    embed_tokens = inner.embed_tokens
    norm = inner.norm

    # Build cache
    cache = Qwen3_5DynamicCache(text_config)

    # We need rotary embeddings for attention layers
    rotary_emb = inner.rotary_emb if hasattr(inner, 'rotary_emb') else None

    # Use -1 as key for embedding output (input to layer 0)
    results = {idx: [] for idx in layer_indices}
    results[-1] = []  # embeddings
    logits_per_step = []

    all_ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
    n_prompt = len(all_ids)

    print(f"[Ref] Prefill {n_prompt} tokens, then decode {num_tokens} tokens...")

    def run_one_token(token_id, pos):
        inp = torch.tensor([[token_id]], dtype=torch.long)
        cache_position = torch.tensor([pos], dtype=torch.long)

        with torch.no_grad():
            hidden_states = embed_tokens(inp)
            results[-1].append(hidden_states.detach().clone())

            position_ids = cache_position.unsqueeze(0)
            if rotary_emb is not None:
                position_embeddings = rotary_emb(hidden_states, position_ids)
            else:
                position_embeddings = (None, None)

            for i, layer_mod in enumerate(layers):
                hidden_states = layer_mod(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    past_key_values=cache,
                    cache_position=cache_position,
                )
                if i in layer_indices:
                    results[i].append(hidden_states.detach().clone())

            hidden_states = norm(hidden_states)

        return hidden_states

    # Prefill
    generated = []
    for step, token_id in enumerate(all_ids):
        h = run_one_token(token_id, step)
        if step == n_prompt - 1:
            with torch.no_grad():
                lm_head = model.lm_head if hasattr(model, 'lm_head') else model.model.lm_head
                logits = lm_head(h)
                logits_per_step.append(logits.detach().clone())
                generated.append(logits.argmax(dim=-1).item())

    # Decode
    for di in range(num_tokens - 1):
        h = run_one_token(generated[-1], n_prompt + di)
        with torch.no_grad():
            lm_head = model.lm_head if hasattr(model, 'lm_head') else model.model.lm_head
            logits = lm_head(h)
            logits_per_step.append(logits.detach().clone())
            generated.append(logits.argmax(dim=-1).item())

    return results, logits_per_step, generated


# ============ Fused Step-by-step ============

def fused_step_by_step(fused_layers, fused_layer_indices, embed_weights, input_ids,
                       config, num_tokens, ref_inter_layer_states=None):
    """Run fused DeltaNet layers step-by-step.

    For non-DeltaNet layers (full attention), we use the reference hidden states
    to feed through, so we can isolate DeltaNet divergence.

    fused_layers: dict {layer_idx: FusedDeltaNetLayer}
    fused_layer_indices: list of DeltaNet layer indices
    ref_inter_layer_states: dict {layer_idx: [hidden_state_per_token]} from reference
                           Used to feed correct inputs to isolated DeltaNet layers.
    """
    lin_num_key_heads = config["linear_num_key_heads"]
    lin_key_dim = config["linear_key_head_dim"]
    lin_num_val_heads = config["linear_num_value_heads"]
    lin_val_dim = config["linear_value_head_dim"]
    lin_qkv_dim = 2 * lin_num_key_heads * lin_key_dim + lin_num_val_heads * lin_val_dim
    conv_kernel = config.get("linear_conv_kernel_dim", 4)

    # Initialize states per layer
    conv_states = {}
    ssm_states = {}
    for idx in fused_layer_indices:
        conv_states[idx] = torch.zeros(lin_qkv_dim, conv_kernel)
        ssm_states[idx] = torch.zeros(lin_num_val_heads, lin_key_dim, lin_val_dim)

    results = {idx: [] for idx in fused_layer_indices}

    all_ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
    n_prompt = len(all_ids)
    total_steps = n_prompt + num_tokens - 1

    print(f"[Fused] Running {total_steps} steps on {len(fused_layer_indices)} DeltaNet layers...")

    for step in range(total_steps):
        for idx in fused_layer_indices:
            # Get input to this layer from reference:
            # - Layer 0 uses embedding output (key -1)
            # - Layer N uses output of layer N-1
            prev_key = idx - 1 if idx > 0 else -1
            if ref_inter_layer_states is not None and prev_key in ref_inter_layer_states:
                if step < len(ref_inter_layer_states[prev_key]):
                    x = ref_inter_layer_states[prev_key][step].squeeze(0)  # [1, hidden]
                else:
                    continue
            else:
                continue

            with torch.no_grad():
                out, new_cs, new_ss = fused_layers[idx](x, conv_states[idx], ssm_states[idx])
                conv_states[idx] = new_cs
                ssm_states[idx] = new_ss
                results[idx].append(out.detach().clone())

    return results


# ============ Main ============

def compare_tensors(name, ref, fused, step):
    """Compare two tensors and print stats."""
    ref_flat = ref.flatten().float()
    fused_flat = fused.flatten().float()
    diff = (ref_flat - fused_flat).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (ref_flat.abs() + 1e-8)).mean().item()

    # Cosine similarity
    cos = F.cosine_similarity(ref_flat.unsqueeze(0), fused_flat.unsqueeze(0)).item()

    status = "OK" if max_diff < 1e-4 else ("WARN" if max_diff < 1e-2 else "FAIL")
    print(f"  [{status}] {name} step={step}: max={max_diff:.6e} mean={mean_diff:.6e} "
          f"rel={rel_diff:.6e} cos={cos:.8f}")
    return max_diff, cos


def main():
    parser = argparse.ArgumentParser(description="Validate fused DeltaNet vs transformers reference")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--tokens", type=int, default=10, help="Number of decode tokens to generate")
    parser.add_argument("--layers", type=str, default="0,1,2",
                        help="Comma-separated DeltaNet layer indices to test (default: 0,1,2)")
    parser.add_argument("--prompt", type=str, default="The capital of France is",
                        help="Test prompt")
    args = parser.parse_args()

    test_layers = [int(x) for x in args.layers.split(",")]

    # Load config
    config_path = os.path.join(args.model, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    tc = config.get("text_config", config)

    num_layers = tc["num_hidden_layers"]
    full_attn_interval = tc.get("full_attention_interval", 4)

    # Determine layer types
    layer_types = []
    for i in range(num_layers):
        if (i + 1) % full_attn_interval == 0:
            layer_types.append("full_attention")
        else:
            layer_types.append("linear_attention")

    dn_layers = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
    print(f"DeltaNet layers: {dn_layers}")
    print(f"Testing layers: {test_layers}")

    # Validate requested layers are DeltaNet
    for idx in test_layers:
        if idx >= num_layers:
            print(f"ERROR: Layer {idx} >= num_layers ({num_layers})")
            sys.exit(1)
        if layer_types[idx] != "linear_attention":
            print(f"WARNING: Layer {idx} is full_attention, not DeltaNet — skipping")
            test_layers.remove(idx)

    # Load reference model
    ref_model, tokenizer = load_reference_model(args.model)

    # Tokenize
    input_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    print(f"\nPrompt: {args.prompt!r}")
    print(f"Token IDs ({len(input_ids)}): {input_ids}")

    # Which layers' outputs we need from the reference (test layers + their predecessors)
    capture_layers = set(test_layers)
    for idx in test_layers:
        if idx > 0:
            capture_layers.add(idx - 1)

    # Run reference model step-by-step
    ref_results, ref_logits, ref_generated = reference_step_by_step(
        ref_model, input_ids, capture_layers, args.tokens
    )

    decoded = tokenizer.decode(ref_generated, skip_special_tokens=True)
    print(f"\n[Ref] Generated: {decoded!r}")
    print(f"[Ref] Token IDs: {ref_generated}")

    # Show first few logits
    for i, logits in enumerate(ref_logits[:3]):
        top5 = logits.flatten().topk(5)
        top_tokens = [(tokenizer.decode([idx.item()]), val.item()) for idx, val in zip(top5.indices, top5.values)]
        print(f"[Ref] Step {i} top-5: {top_tokens}")

    # Load fused layers
    print(f"\n--- Loading fused DeltaNet layers ---")
    handles, weight_map = open_safetensors(args.model)
    prefix = detect_prefix(handles, weight_map)

    fused_layers = {}
    for idx in test_layers:
        print(f"  Loading fused layer {idx}...")
        fused_layers[idx] = load_fused_deltanet(handles, weight_map, prefix, idx, tc)
        fused_layers[idx].eval()

    # Run fused layers step-by-step, using reference inputs
    fused_results = fused_step_by_step(
        fused_layers, test_layers, None, input_ids, tc, args.tokens,
        ref_inter_layer_states=ref_results
    )

    # Compare outputs
    print(f"\n{'='*80}")
    print(f"COMPARISON: Fused vs Reference per-layer per-step")
    print(f"{'='*80}")

    for idx in test_layers:
        print(f"\n--- Layer {idx} (DeltaNet) ---")
        ref_list = ref_results.get(idx, [])
        fused_list = fused_results.get(idx, [])
        n_compare = min(len(ref_list), len(fused_list))

        if n_compare == 0:
            print(f"  No outputs to compare (ref={len(ref_list)}, fused={len(fused_list)})")
            continue

        max_diffs = []
        cosines = []
        for step in range(n_compare):
            md, cos = compare_tensors(f"L{idx}", ref_list[step], fused_list[step], step)
            max_diffs.append(md)
            cosines.append(cos)

        # Summary
        print(f"\n  Summary L{idx}: {n_compare} steps")
        print(f"    max_diff: min={min(max_diffs):.6e} max={max(max_diffs):.6e}")
        print(f"    cosine:   min={min(cosines):.8f} max={max(cosines):.8f}")

        # Find first divergence point
        for step, md in enumerate(max_diffs):
            if md > 1e-4:
                print(f"    ** First significant divergence at step {step} (max_diff={md:.6e})")
                break
        else:
            print(f"    All steps within tolerance (max_diff < 1e-4)")


if __name__ == "__main__":
    main()
