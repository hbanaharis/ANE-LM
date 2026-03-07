#!/usr/bin/env python3
"""Detailed single-step comparison of fused DeltaNet vs transformers reference.

Compares intermediate values at every stage within a single DeltaNet layer
to pinpoint exactly where the divergence occurs.

Usage:
    cd /Users/hb/projects/ANE-LM
    python scripts/validate_deltanet_detailed.py --model models/Qwen3.5-4B
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


def l2norm(x, dim=-1, eps=1e-6):
    """Match transformers l2norm exactly."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


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


def compare(name, a, b, detail=True):
    """Compare two tensors and print stats."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    diff = (a_flat - b_flat).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()

    status = "OK" if max_diff < 1e-5 else ("WARN" if max_diff < 1e-3 else "FAIL")
    print(f"  [{status}] {name}: max={max_diff:.6e} mean={mean_diff:.6e} cos={cos:.8f}"
          f"  shape={list(a.shape)}")
    if detail and max_diff > 1e-3:
        # Show where the biggest diff is
        idx = diff.argmax().item()
        print(f"         worst@{idx}: ref={a_flat[idx].item():.6f} fused={b_flat[idx].item():.6f}")
        # Show a few values
        print(f"         ref[:5]  = {a_flat[:5].tolist()}")
        print(f"         fused[:5]= {b_flat[:5].tolist()}")
    return max_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--layer", type=int, default=0, help="DeltaNet layer index to test")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(args.model, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    tc = config.get("text_config", config)

    hidden = tc["hidden_size"]
    num_layers = tc["num_hidden_layers"]
    full_attn_interval = tc.get("full_attention_interval", 4)
    lin_num_key_heads = tc["linear_num_key_heads"]
    lin_key_dim = tc["linear_key_head_dim"]
    lin_num_val_heads = tc["linear_num_value_heads"]
    lin_val_dim = tc["linear_value_head_dim"]
    lin_qkv_dim = 2 * lin_num_key_heads * lin_key_dim + lin_num_val_heads * lin_val_dim
    lin_total_val = lin_num_val_heads * lin_val_dim
    conv_kernel = tc.get("linear_conv_kernel_dim", 4)
    eps = tc.get("rms_norm_eps", 1e-6)
    head_ratio = lin_num_val_heads // lin_num_key_heads

    layer_idx = args.layer
    # Verify it's a DeltaNet layer
    if (layer_idx + 1) % full_attn_interval == 0:
        print(f"ERROR: Layer {layer_idx} is full_attention, not DeltaNet")
        sys.exit(1)

    print(f"Testing DeltaNet layer {layer_idx}")
    print(f"  hidden={hidden}, lin_qkv_dim={lin_qkv_dim}, lin_total_val={lin_total_val}")
    print(f"  key_heads={lin_num_key_heads}, val_heads={lin_num_val_heads}")
    print(f"  key_dim={lin_key_dim}, val_dim={lin_val_dim}, head_ratio={head_ratio}")
    print(f"  conv_kernel={conv_kernel}")

    # Load weights
    handles, weight_map = open_safetensors(args.model)

    # Detect prefix
    for pfx in ["model.language_model.", "language_model.model.", "model.", ""]:
        test_key = f"{pfx}layers.0.input_layernorm.weight"
        if test_key in weight_map:
            prefix = pfx
            break
    else:
        raise ValueError("Cannot detect weight prefix")

    wpfx = f"{prefix}layers.{layer_idx}"
    print(f"  weight prefix: '{wpfx}'")

    # Load individual weight tensors
    input_ln_w = get_tensor(handles, weight_map, f"{wpfx}.input_layernorm.weight")
    qkv_w = get_tensor(handles, weight_map, f"{wpfx}.linear_attn.in_proj_qkv.weight")
    z_w = get_tensor(handles, weight_map, f"{wpfx}.linear_attn.in_proj_z.weight")
    a_w = get_tensor(handles, weight_map, f"{wpfx}.linear_attn.in_proj_a.weight")
    b_w = get_tensor(handles, weight_map, f"{wpfx}.linear_attn.in_proj_b.weight")
    conv1d_w = get_tensor(handles, weight_map, f"{wpfx}.linear_attn.conv1d.weight")  # [C, 1, K]
    A_log = get_tensor(handles, weight_map, f"{wpfx}.linear_attn.A_log")
    dt_bias = get_tensor(handles, weight_map, f"{wpfx}.linear_attn.dt_bias")
    norm_w = get_tensor(handles, weight_map, f"{wpfx}.linear_attn.norm.weight")
    o_proj_w = get_tensor(handles, weight_map, f"{wpfx}.linear_attn.out_proj.weight")
    post_ln_w = get_tensor(handles, weight_map, f"{wpfx}.post_attention_layernorm.weight")
    gate_proj_w = get_tensor(handles, weight_map, f"{wpfx}.mlp.gate_proj.weight")
    up_proj_w = get_tensor(handles, weight_map, f"{wpfx}.mlp.up_proj.weight")
    down_proj_w = get_tensor(handles, weight_map, f"{wpfx}.mlp.down_proj.weight")

    print(f"  qkv_w: {qkv_w.shape}, z_w: {z_w.shape}, a_w: {a_w.shape}, b_w: {b_w.shape}")
    print(f"  conv1d_w: {conv1d_w.shape}")
    print(f"  fused_w would be: [{qkv_w.shape[0]}+{z_w.shape[0]}+{a_w.shape[0]}+{b_w.shape[0]}={qkv_w.shape[0]+z_w.shape[0]+a_w.shape[0]+b_w.shape[0]}, {hidden}]")

    # Create a random input (like an embedding)
    torch.manual_seed(42)
    x = torch.randn(1, hidden)

    # Zero initial states
    conv_state = torch.zeros(lin_qkv_dim, conv_kernel)
    ssm_state = torch.zeros(lin_num_val_heads, lin_key_dim, lin_val_dim)

    print(f"\n{'='*80}")
    print("STEP-BY-STEP COMPARISON (zero initial state, random input)")
    print(f"{'='*80}")

    # ============ 1. Input LayerNorm ============
    print("\n--- 1. Input LayerNorm ---")
    def rmsnorm(x, w, e):
        """Qwen3_5 RMSNorm: weight applied as (1 + weight), NOT just weight."""
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + e) * (1.0 + w)

    ref_normed = rmsnorm(x, input_ln_w, eps)

    # Fused: same thing
    fused_normed = rmsnorm(x, input_ln_w, eps)
    compare("layernorm", ref_normed, fused_normed)

    # ============ 2. Projections ============
    print("\n--- 2. Projections (separate vs fused) ---")
    # Reference: separate projections
    ref_qkv = F.linear(ref_normed, qkv_w)  # [1, qkv_dim]
    ref_z = F.linear(ref_normed, z_w)       # [1, total_val]
    ref_b = F.linear(ref_normed, b_w)       # [1, num_v_heads]
    ref_a = F.linear(ref_normed, a_w)       # [1, num_v_heads]

    # Fused: single projection
    fused_w = torch.cat([qkv_w, z_w, a_w, b_w], dim=0)
    fused_proj = F.linear(fused_normed, fused_w)
    fused_qkv = fused_proj[:, :lin_qkv_dim]
    fused_z = fused_proj[:, lin_qkv_dim:lin_qkv_dim + lin_total_val]
    fused_ab = fused_proj[:, lin_qkv_dim + lin_total_val:]
    fused_a = fused_ab[:, :lin_num_val_heads]
    fused_b = fused_ab[:, lin_num_val_heads:]

    compare("qkv", ref_qkv, fused_qkv)
    compare("z", ref_z, fused_z)
    compare("a", ref_a, fused_a)
    compare("b", ref_b, fused_b)

    # ============ 3. Conv1d ============
    print("\n--- 3. Conv1d update ---")
    # Reference: conv state management
    # On first token with empty state, the ref does:
    #   mixed_qkv = mixed_qkv.transpose(1,2)  → [1, qkv_dim, 1]
    #   conv_state = F.pad(mixed_qkv, (conv_kernel - 1, 0))  → [1, qkv_dim, conv_kernel]
    #   mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :1])
    # But for update (decode) path:
    #   mixed_qkv: [1, qkv_dim, 1]
    #   torch_causal_conv1d_update: cat([conv_state, x], dim=-1) → [1, C, K+1]
    #   F.conv1d → [1, C, 2], take last
    conv_w_squeezed = conv1d_w.squeeze(1)  # [C, K]

    # Reference conv update (torch_causal_conv1d_update with empty state)
    ref_mixed = ref_qkv.transpose(0, 1).unsqueeze(0)  # [1, qkv_dim, 1]
    ref_conv_state = torch.zeros(1, lin_qkv_dim, conv_kernel)
    hidden_new = torch.cat([ref_conv_state, ref_mixed], dim=-1)  # [1, C, K+1]
    ref_new_conv_state = hidden_new[:, :, -conv_kernel:]  # [1, C, K]
    ref_conv_out = F.conv1d(hidden_new, conv1d_w, bias=None, padding=0, groups=lin_qkv_dim)
    ref_conv_out = F.silu(ref_conv_out[:, :, -1:])  # [1, C, 1]
    ref_conv_out = ref_conv_out.squeeze(0).squeeze(-1)  # [C]

    # Fused conv update
    fused_mixed = fused_qkv.squeeze(0)  # [qkv_dim]
    fused_new_conv_state = torch.cat([conv_state[:, 1:], fused_mixed.unsqueeze(1)], dim=1)
    fused_conv_out = (fused_new_conv_state * conv_w_squeezed).sum(dim=1)
    fused_conv_out = F.silu(fused_conv_out)

    compare("conv_state", ref_new_conv_state.squeeze(0), fused_new_conv_state)
    compare("conv_out", ref_conv_out, fused_conv_out)

    # ============ 4. Q/K/V split ============
    print("\n--- 4. Q/K/V split ---")
    key_total = lin_num_key_heads * lin_key_dim
    ref_q = ref_conv_out[:key_total].view(lin_num_key_heads, lin_key_dim)
    ref_k = ref_conv_out[key_total:2*key_total].view(lin_num_key_heads, lin_key_dim)
    ref_v = ref_conv_out[2*key_total:].view(lin_num_val_heads, lin_val_dim)

    fused_q = fused_conv_out[:key_total].view(lin_num_key_heads, lin_key_dim)
    fused_k = fused_conv_out[key_total:2*key_total].view(lin_num_key_heads, lin_key_dim)
    fused_v = fused_conv_out[2*key_total:].view(lin_num_val_heads, lin_val_dim)

    compare("q_raw", ref_q, fused_q, detail=False)
    compare("k_raw", ref_k, fused_k, detail=False)
    compare("v_raw", ref_v, fused_v, detail=False)

    # ============ 5. Repeat interleave + L2 norm + scale ============
    print("\n--- 5. Repeat interleave + L2 norm + scale ---")
    ref_q = ref_q.repeat_interleave(head_ratio, dim=0)
    ref_k = ref_k.repeat_interleave(head_ratio, dim=0)
    ref_q = l2norm(ref_q, dim=-1, eps=1e-6)
    ref_k = l2norm(ref_k, dim=-1, eps=1e-6)
    scale = 1.0 / (lin_key_dim ** 0.5)
    ref_q = ref_q * scale

    fused_q = fused_q.repeat_interleave(head_ratio, dim=0)
    fused_k = fused_k.repeat_interleave(head_ratio, dim=0)
    fused_q = l2norm(fused_q, dim=-1, eps=1e-6)
    fused_k = l2norm(fused_k, dim=-1, eps=1e-6)
    fused_q = fused_q * scale

    compare("q_normed_scaled", ref_q, fused_q, detail=False)
    compare("k_normed", ref_k, fused_k, detail=False)

    # ============ 6. Beta and decay ============
    print("\n--- 6. Beta and decay ---")
    ref_beta = torch.sigmoid(ref_b).squeeze(0)
    ref_g = -A_log.float().exp() * F.softplus(ref_a.float().squeeze(0) + dt_bias)
    ref_decay = torch.exp(ref_g)

    fused_beta = torch.sigmoid(fused_b).squeeze(0)
    fused_g = -A_log.float().exp() * F.softplus(fused_a.float().squeeze(0) + dt_bias)
    fused_decay = torch.exp(fused_g)

    compare("beta", ref_beta, fused_beta, detail=False)
    compare("decay", ref_decay, fused_decay, detail=False)

    # ============ 7. SSM step ============
    print("\n--- 7. SSM step (zero initial state) ---")
    # Reference (torch_recurrent_gated_delta_rule style)
    ref_decayed = ref_decay.view(-1, 1, 1) * ssm_state
    ref_Sk = torch.einsum('hkv,hk->hv', ref_decayed, ref_k)
    ref_delta = (ref_v - ref_Sk) * ref_beta.view(-1, 1)
    ref_outer = ref_k.unsqueeze(2) * ref_delta.unsqueeze(1)
    ref_new_ssm = ref_decayed + ref_outer
    ref_y = torch.einsum('hkv,hk->hv', ref_new_ssm, ref_q)

    fused_decayed = fused_decay.view(-1, 1, 1) * ssm_state
    fused_Sk = torch.einsum('hkv,hk->hv', fused_decayed, fused_k)
    fused_delta = (fused_v - fused_Sk) * fused_beta.view(-1, 1)
    fused_outer = fused_k.unsqueeze(2) * fused_delta.unsqueeze(1)
    fused_new_ssm = fused_decayed + fused_outer
    fused_y = torch.einsum('hkv,hk->hv', fused_new_ssm, fused_q)

    compare("Sk", ref_Sk, fused_Sk, detail=False)
    compare("delta", ref_delta, fused_delta, detail=False)
    compare("new_ssm_state", ref_new_ssm, fused_new_ssm, detail=False)
    compare("y (SSM output)", ref_y, fused_y, detail=False)

    # ============ 8. Gated RMSNorm ============
    print("\n--- 8. Gated RMSNorm ---")
    ref_y32 = ref_y.float()
    var_ref = ref_y32.pow(2).mean(dim=-1, keepdim=True)
    ref_y_norm = ref_y32 * torch.rsqrt(var_ref + 1e-6)
    ref_y_norm = norm_w.unsqueeze(0) * ref_y_norm
    ref_z_reshaped = ref_z.view(lin_num_val_heads, lin_val_dim)
    ref_y_gated = (ref_y_norm * F.silu(ref_z_reshaped.float())).reshape(-1)

    fused_y32 = fused_y.float()
    var_fused = fused_y32.pow(2).mean(dim=-1, keepdim=True)
    fused_y_norm = fused_y32 * torch.rsqrt(var_fused + 1e-6)
    fused_y_norm = norm_w.unsqueeze(0) * fused_y_norm
    fused_z_reshaped = fused_z.view(lin_num_val_heads, lin_val_dim)
    fused_y_gated = (fused_y_norm * F.silu(fused_z_reshaped.float())).reshape(-1)

    compare("y_normed", ref_y_norm, fused_y_norm, detail=False)
    compare("y_gated", ref_y_gated, fused_y_gated, detail=False)

    # ============ 9. Output projection + residual ============
    print("\n--- 9. Output projection + residual ---")
    ref_attn_out = F.linear(ref_y_gated.unsqueeze(0), o_proj_w)
    ref_out1 = x + ref_attn_out

    fused_attn_out = F.linear(fused_y_gated.unsqueeze(0), o_proj_w)
    fused_out1 = x + fused_attn_out

    compare("attn_out (o_proj)", ref_attn_out, fused_attn_out, detail=False)
    compare("after_residual_1", ref_out1, fused_out1, detail=False)

    # ============ 10. FFN ============
    print("\n--- 10. Post-attn norm + FFN + residual ---")
    ref_post_norm = rmsnorm(ref_out1, post_ln_w, eps)
    ref_ffn = F.linear(F.silu(F.linear(ref_post_norm, gate_proj_w)) * F.linear(ref_post_norm, up_proj_w), down_proj_w)
    ref_final = ref_out1 + ref_ffn

    fused_post_norm = rmsnorm(fused_out1, post_ln_w, eps)
    fused_ffn = F.linear(F.silu(F.linear(fused_post_norm, gate_proj_w)) * F.linear(fused_post_norm, up_proj_w), down_proj_w)
    fused_final = fused_out1 + fused_ffn

    compare("post_attn_norm", ref_post_norm, fused_post_norm, detail=False)
    compare("ffn_out", ref_ffn, fused_ffn, detail=False)
    compare("FINAL layer output", ref_final, fused_final, detail=False)

    # ============ Now compare against the actual transformers model ============
    print(f"\n{'='*80}")
    print("COMPARISON VS TRANSFORMERS MODEL (actual model forward)")
    print(f"{'='*80}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\nLoading transformers model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Get the specific layer
    text_model = model.model if hasattr(model, 'model') else model
    if hasattr(text_model, 'language_model'):
        text_model = text_model.language_model
    if hasattr(text_model, 'model'):
        inner = text_model.model
    else:
        inner = text_model

    layers = inner.layers
    embed_tokens = inner.embed_tokens
    final_norm = inner.norm
    text_config = inner.config if hasattr(inner, 'config') else model.config
    if hasattr(text_config, 'text_config'):
        text_config = text_config.text_config

    target_layer = layers[layer_idx]
    print(f"Target layer type: {type(target_layer).__name__}")
    print(f"Layer type attr: {target_layer.layer_type}")

    # Direct weight comparison: model weights vs safetensor weights
    print("\n--- Direct weight comparison ---")
    model_ln_w = target_layer.input_layernorm.weight.data.float()
    compare("input_layernorm.weight", model_ln_w, input_ln_w, detail=True)

    model_qkv_w = target_layer.linear_attn.in_proj_qkv.weight.data.float()
    compare("in_proj_qkv.weight", model_qkv_w, qkv_w, detail=True)

    model_embed_w = embed_tokens.weight.data.float()
    print(f"  embed shape: model={model_embed_w.shape}")
    # Compare a slice
    compare("embed_tokens.weight[0:5]", model_embed_w[:5],
            get_tensor(handles, weight_map, f"{prefix}embed_tokens.weight")[:5], detail=True)

    # Tokenize a test prompt
    prompt = "Hello"
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"Prompt: {prompt!r}, tokens: {input_ids}")

    # Get embedding for first token
    tok_tensor = torch.tensor([[input_ids[0]]], dtype=torch.long)
    with torch.no_grad():
        embed_out = embed_tokens(tok_tensor)  # [1, 1, hidden]

    print(f"\nembedding shape: {embed_out.shape}")
    print(f"  embed_out[:5] = {embed_out.flatten()[:5].tolist()}")
    embed_1d = embed_out.squeeze(0)  # [1, hidden]

    # Manual embedding lookup from safetensors
    embed_w_st = get_tensor(handles, weight_map, f"{prefix}embed_tokens.weight")
    manual_embed = embed_w_st[input_ids[0]:input_ids[0]+1]  # [1, hidden]
    print(f"  manual_embed[:5] = {manual_embed.flatten()[:5].tolist()}")
    compare("embedding lookup", embed_1d, manual_embed)

    # --- Run through transformers layer ---
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
    cache = Qwen3_5DynamicCache(text_config)
    cache_position = torch.tensor([0], dtype=torch.long)
    position_ids = cache_position.unsqueeze(0)
    rotary_emb = inner.rotary_emb if hasattr(inner, 'rotary_emb') else None
    if rotary_emb is not None:
        position_embeddings = rotary_emb(embed_out, position_ids)
    else:
        position_embeddings = (None, None)

    print(f"\nModel class: {type(model).__name__}")
    print(f"inner class: {type(inner).__name__}")

    # Hook into the layer to capture intermediates
    intermediates = {}

    # Pre-hook to capture actual input to layernorm
    def ln_pre_hook(module, input):
        intermediates["input_layernorm_INPUT"] = input[0].detach().clone()
    hooks_pre = [target_layer.input_layernorm.register_forward_pre_hook(ln_pre_hook)]

    # Also capture layer's own input
    def layer_pre_hook(module, input):
        intermediates["layer_INPUT"] = input[0].detach().clone()
    hooks_pre.append(target_layer.register_forward_pre_hook(layer_pre_hook))

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                intermediates[name] = tuple(o.detach().clone() if isinstance(o, torch.Tensor) else o for o in output)
            else:
                intermediates[name] = output.detach().clone()
        return hook_fn

    # Register hooks on the linear_attn sub-module
    linear_attn = target_layer.linear_attn
    hooks = []
    hooks.append(linear_attn.in_proj_qkv.register_forward_hook(make_hook("in_proj_qkv")))
    hooks.append(linear_attn.in_proj_z.register_forward_hook(make_hook("in_proj_z")))
    hooks.append(linear_attn.in_proj_a.register_forward_hook(make_hook("in_proj_a")))
    hooks.append(linear_attn.in_proj_b.register_forward_hook(make_hook("in_proj_b")))
    hooks.append(linear_attn.out_proj.register_forward_hook(make_hook("out_proj")))
    hooks.append(linear_attn.norm.register_forward_hook(make_hook("norm")))
    hooks.append(target_layer.input_layernorm.register_forward_hook(make_hook("input_layernorm")))

    with torch.no_grad():
        transformers_out = target_layer(
            embed_out,
            position_embeddings=position_embeddings,
            past_key_values=cache,
            cache_position=cache_position,
        )

    for h in hooks:
        h.remove()
    for h in hooks_pre:
        h.remove()

    # Check what the layernorm actually received as input
    if "layer_INPUT" in intermediates:
        print(f"\n  layer_INPUT shape: {intermediates['layer_INPUT'].shape}")
        print(f"  layer_INPUT[:5] = {intermediates['layer_INPUT'].flatten()[:5].tolist()}")
        compare("layer_INPUT vs embed_out", intermediates['layer_INPUT'].squeeze(0), embed_1d)
    if "input_layernorm_INPUT" in intermediates:
        print(f"  ln_INPUT shape: {intermediates['input_layernorm_INPUT'].shape}")
        print(f"  ln_INPUT[:5] = {intermediates['input_layernorm_INPUT'].flatten()[:5].tolist()}")
        compare("ln_INPUT vs embed_out", intermediates['input_layernorm_INPUT'].squeeze(0), embed_1d)

    print(f"\ntransformers layer output shape: {transformers_out.shape}")

    # Now run the fused layer on the same input
    print("\n--- Comparing intermediates ---")

    # 1. Input layernorm
    fused_ln = rmsnorm(embed_1d, input_ln_w, eps)

    # DIRECT call to the model's layernorm (outside layer forward)
    with torch.no_grad():
        direct_ln = target_layer.input_layernorm(embed_out).squeeze(0)
    compare("DIRECT layernorm call", direct_ln, fused_ln)
    print(f"  direct_ln[:5] = {direct_ln.flatten()[:5].tolist()}")

    if "input_layernorm" in intermediates:
        ref_ln = intermediates["input_layernorm"].squeeze(0)  # [1, hidden]
        compare("hook vs direct", ref_ln, direct_ln)
        compare("hook vs fused", ref_ln, fused_ln)

        # Check the RMS of the input
        rms_embed = embed_1d.pow(2).mean(-1).sqrt().item()
        print(f"  RMS of embed_1d: {rms_embed:.8f}")
        print(f"  eps: {eps}")

        # Inspect the model's layernorm
        ln_mod = target_layer.input_layernorm
        print(f"  layernorm class: {type(ln_mod).__name__}")
        print(f"  layernorm attrs: {[a for a in dir(ln_mod) if not a.startswith('_')]}")
        # Try different eps attribute names
        for attr in ['variance_epsilon', 'eps', 'epsilon', 'norm_eps']:
            if hasattr(ln_mod, attr):
                print(f"  layernorm {attr}: {getattr(ln_mod, attr)}")
        print(f"  layernorm weight shape: {ln_mod.weight.shape}")
        print(f"  layernorm weight min/max: {ln_mod.weight.min().item():.6f} / {ln_mod.weight.max().item():.6f}")

        # Check source code of the forward method
        import inspect
        try:
            src = inspect.getsource(type(ln_mod).forward)
            print(f"  forward source:\n{src[:500]}")
        except:
            print(f"  Could not get forward source")

        # Check if there's a compiled kernel
        print(f"  forward method: {type(ln_mod).forward}")

        # Test with 3D input (matching model)
        x_3d = embed_out.clone().float()  # [1, 1, 2560]
        var_3d = x_3d.pow(2).mean(-1, keepdim=True)
        normed_3d = x_3d * torch.rsqrt(var_3d + eps)
        weighted_3d = ln_mod.weight.float() * normed_3d
        print(f"  manual 3D normed[:5] = {weighted_3d.flatten()[:5].tolist()}")

        # Manual test: compute rmsnorm step by step with explicit values
        x_test = embed_out.clone().float()  # [1, 1, 2560]
        var_test = x_test.pow(2).mean(-1, keepdim=True)
        print(f"  manual var: {var_test.item():.10e}")
        normed_test = x_test * torch.rsqrt(var_test + eps)
        print(f"  manual normed[:5] = {normed_test.flatten()[:5].tolist()}")
        weighted_test = ln_mod.weight * normed_test
        print(f"  manual weighted[:5] = {weighted_test.flatten()[:5].tolist()}")
        print(f"  model  output[:5]   = {direct_ln.flatten()[:5].tolist()}")

    # 2. Projections
    if "in_proj_qkv" in intermediates:
        ref_qkv_out = intermediates["in_proj_qkv"]  # [1, 1, qkv_dim]
        fused_qkv_out = F.linear(fused_ln, qkv_w)
        compare("in_proj_qkv", ref_qkv_out.reshape(1, -1), fused_qkv_out)

    if "in_proj_z" in intermediates:
        ref_z_out = intermediates["in_proj_z"]
        fused_z_out = F.linear(fused_ln, z_w)
        compare("in_proj_z", ref_z_out.reshape(1, -1), fused_z_out)

    if "in_proj_a" in intermediates:
        ref_a_out = intermediates["in_proj_a"]
        fused_a_out = F.linear(fused_ln, a_w)
        compare("in_proj_a", ref_a_out.reshape(1, -1), fused_a_out)

    if "in_proj_b" in intermediates:
        ref_b_out = intermediates["in_proj_b"]
        fused_b_out = F.linear(fused_ln, b_w)
        compare("in_proj_b", ref_b_out.reshape(1, -1), fused_b_out)

    # 3. Conv state: on first token, transformers stores the state and runs conv
    # Check the conv state after first token
    conv_state_ref = cache.conv_states[layer_idx]
    print(f"\n  ref conv_state shape: {conv_state_ref.shape if conv_state_ref is not None else 'None'}")

    # The ref stores conv_state = F.pad(mixed_qkv_transposed, (K - seq_len, 0))
    # For seq_len=1: pad (K-1) zeros on left + 1 new value = [1, C, K]
    if conv_state_ref is not None:
        # Fused conv state after first token: zeros shifted, new value appended
        fused_qkv_val = F.linear(fused_ln, qkv_w).squeeze(0)  # [qkv_dim]
        fused_cs = torch.zeros(lin_qkv_dim, conv_kernel)
        fused_cs = torch.cat([fused_cs[:, 1:], fused_qkv_val.unsqueeze(1)], dim=1)
        # ref_conv_state: [1, C, K] — the last position is the new input
        compare("conv_state", conv_state_ref.squeeze(0), fused_cs)

    # 4. Check the recurrent state
    ssm_state_ref = cache.recurrent_states[layer_idx]
    print(f"\n  ref ssm_state shape: {ssm_state_ref.shape if ssm_state_ref is not None else 'None'}")

    # 5. Full layer output comparison
    print(f"\n--- Final layer output ---")
    # Run the fused DeltaNetLayer (from export_hybrid_chunks.py logic)
    # Reconstruct the fused layer forward manually

    # Start from embedding
    x_in = embed_1d  # [1, hidden]
    residual = x_in
    h = rmsnorm(x_in, input_ln_w, eps)

    # Fused projection
    fused_w_cat = torch.cat([qkv_w, z_w, a_w, b_w], dim=0)
    proj = F.linear(h, fused_w_cat)
    mixed_qkv_f = proj[:, :lin_qkv_dim]
    z_f = proj[:, lin_qkv_dim:lin_qkv_dim + lin_total_val]
    ab_f = proj[:, lin_qkv_dim + lin_total_val:]
    a_f = ab_f[:, :lin_num_val_heads]
    b_f = ab_f[:, lin_num_val_heads:]

    # Conv (zero initial state)
    cs_init = torch.zeros(lin_qkv_dim, conv_kernel)
    new_cs = torch.cat([cs_init[:, 1:], mixed_qkv_f.squeeze(0).unsqueeze(1)], dim=1)
    conv_out_f = (new_cs * conv_w_squeezed).sum(dim=1)
    conv_out_f = F.silu(conv_out_f)

    # Q/K/V split
    kt = lin_num_key_heads * lin_key_dim
    q_f = conv_out_f[:kt].view(lin_num_key_heads, lin_key_dim)
    k_f = conv_out_f[kt:2*kt].view(lin_num_key_heads, lin_key_dim)
    v_f = conv_out_f[2*kt:].view(lin_num_val_heads, lin_val_dim)

    q_f = q_f.repeat_interleave(head_ratio, dim=0)
    k_f = k_f.repeat_interleave(head_ratio, dim=0)
    q_f = l2norm(q_f, dim=-1, eps=1e-6)
    k_f = l2norm(k_f, dim=-1, eps=1e-6)
    q_f = q_f * (1.0 / (lin_key_dim ** 0.5))

    beta_f = torch.sigmoid(b_f).squeeze(0)
    g_f = -A_log.float().exp() * F.softplus(a_f.float().squeeze(0) + dt_bias)
    decay_f = torch.exp(g_f)

    ss_init = torch.zeros(lin_num_val_heads, lin_key_dim, lin_val_dim)
    decayed_f = decay_f.view(-1, 1, 1) * ss_init
    Sk_f = torch.einsum('hkv,hk->hv', decayed_f, k_f)
    delta_f = (v_f - Sk_f) * beta_f.view(-1, 1)
    outer_f = k_f.unsqueeze(2) * delta_f.unsqueeze(1)
    new_ss_f = decayed_f + outer_f
    y_f = torch.einsum('hkv,hk->hv', new_ss_f, q_f)

    # Gated RMSNorm
    y_f32 = y_f.float()
    var_f = y_f32.pow(2).mean(dim=-1, keepdim=True)
    y_norm_f = y_f32 * torch.rsqrt(var_f + 1e-6)
    y_norm_f = norm_w.unsqueeze(0) * y_norm_f
    z_rsh = z_f.view(lin_num_val_heads, lin_val_dim)
    y_gated_f = (y_norm_f * F.silu(z_rsh.float())).reshape(-1)

    # O proj + residual
    attn_out_f = F.linear(y_gated_f.unsqueeze(0), o_proj_w)
    out1_f = residual + attn_out_f

    # FFN
    post_norm_f = rmsnorm(out1_f, post_ln_w, eps)
    ffn_f = F.linear(F.silu(F.linear(post_norm_f, gate_proj_w)) * F.linear(post_norm_f, up_proj_w), down_proj_w)
    final_f = out1_f + ffn_f

    compare("FUSED vs TRANSFORMERS final", transformers_out.squeeze(0), final_f)

    # Also check if the SSM states match
    if ssm_state_ref is not None:
        compare("SSM state", ssm_state_ref.squeeze(0), new_ss_f)

    # Print ref SSM state stats for debugging
    if ssm_state_ref is not None:
        ss = ssm_state_ref.squeeze(0)
        print(f"\n  ref ssm_state: min={ss.min():.6f} max={ss.max():.6f} mean={ss.mean():.6f}")
        print(f"  fused ssm_state: min={new_ss_f.min():.6f} max={new_ss_f.max():.6f} mean={new_ss_f.mean():.6f}")


if __name__ == "__main__":
    main()
