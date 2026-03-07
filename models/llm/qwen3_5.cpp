#include "qwen3_5.h"
#include "../../core/cpu_ops.h"
#include <cmath>
#include <fstream>
#include <sys/stat.h>
#include <mach/mach_time.h>

namespace ane_lm {

// Convert mach_absolute_time ticks to microseconds
static double to_us(uint64_t elapsed) {
    static mach_timebase_info_data_t info = {};
    if (info.denom == 0) mach_timebase_info(&info);
    return (double)elapsed * info.numer / info.denom / 1000.0;
}

using json = nlohmann::json;

// --- Qwen35Args::from_json ---

Qwen35Args Qwen35Args::from_json(const json& j) {
    Qwen35Args args;

    // Parse text_config if present, otherwise read from top level
    const json& tc = j.contains("text_config") ? j["text_config"] : j;

    args.hidden_size = tc.value("hidden_size", args.hidden_size);
    args.num_hidden_layers = tc.value("num_hidden_layers", args.num_hidden_layers);
    args.num_attention_heads = tc.value("num_attention_heads", args.num_attention_heads);
    args.num_key_value_heads = tc.value("num_key_value_heads", args.num_key_value_heads);
    args.head_dim = tc.value("head_dim", args.head_dim);
    args.intermediate_size = tc.value("intermediate_size", args.intermediate_size);
    args.vocab_size = tc.value("vocab_size", args.vocab_size);
    args.full_attention_interval = tc.value("full_attention_interval", args.full_attention_interval);
    args.rms_norm_eps = tc.value("rms_norm_eps", args.rms_norm_eps);
    args.tie_word_embeddings = tc.value("tie_word_embeddings", j.value("tie_word_embeddings", args.tie_word_embeddings));
    args.attn_output_gate = tc.value("attn_output_gate", args.attn_output_gate);
    args.linear_num_key_heads = tc.value("linear_num_key_heads", args.linear_num_key_heads);
    args.linear_key_head_dim = tc.value("linear_key_head_dim", args.linear_key_head_dim);
    args.linear_value_head_dim = tc.value("linear_value_head_dim", args.linear_value_head_dim);
    args.linear_num_value_heads = tc.value("linear_num_value_heads", args.linear_num_value_heads);
    args.linear_conv_kernel_dim = tc.value("linear_conv_kernel_dim", args.linear_conv_kernel_dim);

    // RoPE parameters
    if (tc.contains("rope_parameters")) {
        auto& rp = tc["rope_parameters"];
        args.rope_theta = rp.value("rope_theta", args.rope_theta);
        args.partial_rotary_factor = tc.value("partial_rotary_factor", args.partial_rotary_factor);
    } else {
        args.rope_theta = tc.value("rope_theta", args.rope_theta);
        args.partial_rotary_factor = tc.value("partial_rotary_factor", args.partial_rotary_factor);
    }

    // Layer types
    if (tc.contains("layer_types")) {
        for (auto& lt : tc["layer_types"]) {
            std::string s = lt.get<std::string>();
            if (s == "linear_attention") {
                args.layer_types.push_back(LayerType::LinearAttention);
            } else {
                args.layer_types.push_back(LayerType::FullAttention);
            }
        }
    } else {
        for (int i = 0; i < args.num_hidden_layers; i++) {
            if ((i + 1) % args.full_attention_interval == 0) {
                args.layer_types.push_back(LayerType::FullAttention);
            } else {
                args.layer_types.push_back(LayerType::LinearAttention);
            }
        }
    }

    return args;
}

// --- Qwen35Model ---

Qwen35Model::~Qwen35Model() {
    free(embed_tokens_);
    free(final_norm_);
    free(x_);
    free(x_norm_);
    free(logits_);
    free(scratch_qkv_);
    free(scratch_conv_);
    free(scratch_y_);
    free(scratch_attn_);
    free(scratch_tmp_);
    free(rope_cos_);
    free(rope_sin_);
    free(x_batch_);
    free(proj_batch_);
    free(core_batch_);
    free(pre_final_x_);

    // Free MTP weights
    free(mtp_.pre_fc_norm_hidden);
    free(mtp_.pre_fc_norm_embedding);
    free(mtp_.fc);
    free(mtp_.q_proj);
    free(mtp_.k_proj);
    free(mtp_.v_proj);
    free(mtp_.o_proj);
    free(mtp_.q_norm);
    free(mtp_.k_norm);
    free(mtp_.input_layernorm);
    free(mtp_.post_attn_layernorm);
    free(mtp_.gate_proj);
    free(mtp_.up_proj);
    free(mtp_.down_proj);
    free(mtp_.norm);
    free(mtp_kv_cache_.k_cache);
    free(mtp_kv_cache_.v_cache);

    for (int L = 0; L < num_layers_; L++) {
        auto& lw = layers_[L];
        free(lw.input_layernorm);
        free(lw.post_attention_layernorm);

        if (lw.type == LayerType::LinearAttention) {
            free(lw.deltanet.in_proj_a);
            free(lw.deltanet.in_proj_b);
            free(lw.deltanet.conv1d_w);
            free(lw.deltanet.A);
            free(lw.deltanet.dt_bias);
            free(lw.deltanet.norm_w);
        } else {
            free(lw.full_attn.q_norm);
            free(lw.full_attn.k_norm);
        }
    }

    for (int L = 0; L < num_layers_; L++) {
        if (layer_types_[L] == LayerType::FullAttention) {
            free(kv_caches_[L].k_cache);
            free(kv_caches_[L].v_cache);
        }
        if (layer_types_[L] == LayerType::LinearAttention) {
            free(delta_states_[L].ssm_state);
            free(delta_states_[L].conv_state);
        }
        ane_free_layer(&ane_layers_[L]);
    }

    // Free CoreML kernels
    for (auto& cl : coreml_layers_) {
        coreml_free(cl.first_proj);
        coreml_free(cl.o_proj);
        coreml_free(cl.fused_ffn);
        coreml_free(cl.norm_fused_ffn);
        coreml_free_2input(cl.post_attn);
    }
    for (auto* k : coreml_lm_head_) coreml_free(k);

    free_lm_head_ane();
}

void Qwen35Model::reset() {
    for (int L = 0; L < num_layers_; L++) {
        if (layer_types_[L] == LayerType::FullAttention) {
            kv_caches_[L].len = 0;
            kv_caches_[L].start = 0;
            memset(kv_caches_[L].k_cache, 0, (size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_ * sizeof(float));
            memset(kv_caches_[L].v_cache, 0, (size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_ * sizeof(float));
        }
        if (layer_types_[L] == LayerType::LinearAttention) {
            memset(delta_states_[L].ssm_state, 0, (size_t)lin_num_val_heads_ * lin_key_dim_ * lin_val_dim_ * sizeof(float));
            memset(delta_states_[L].conv_state, 0, (size_t)lin_qkv_dim_ * (conv_kernel_ - 1) * sizeof(float));
            delta_states_[L].conv_pos = 0;
        }
    }
    // Reset MTP KV cache
    if (mtp_loaded_) {
        mtp_kv_cache_.len = 0;
        mtp_kv_cache_.start = 0;
        if (mtp_kv_cache_.k_cache)
            memset(mtp_kv_cache_.k_cache, 0, (size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_ * sizeof(float));
        if (mtp_kv_cache_.v_cache)
            memset(mtp_kv_cache_.v_cache, 0, (size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_ * sizeof(float));
    }
}

void Qwen35Model::apply_args(const Qwen35Args& args) {
    hidden_size_ = args.hidden_size;
    intermediate_size_ = args.intermediate_size;
    vocab_size_ = args.vocab_size;
    num_layers_ = args.num_hidden_layers;
    num_q_heads_ = args.num_attention_heads;
    num_kv_heads_ = args.num_key_value_heads;
    head_dim_ = args.head_dim;
    rot_dim_ = args.rotation_dim();
    rope_theta_ = args.rope_theta;
    rms_eps_ = args.rms_norm_eps;
    lin_num_key_heads_ = args.linear_num_key_heads;
    lin_num_val_heads_ = args.linear_num_value_heads;
    lin_key_dim_ = args.linear_key_head_dim;
    lin_val_dim_ = args.linear_value_head_dim;
    lin_total_key_ = lin_num_key_heads_ * lin_key_dim_;
    lin_total_val_ = lin_num_val_heads_ * lin_val_dim_;
    lin_qkv_dim_ = lin_total_key_ * 2 + lin_total_val_;
    conv_kernel_ = args.linear_conv_kernel_dim;
    full_q_dim_ = num_q_heads_ * head_dim_ * 2;
    full_kv_dim_ = num_kv_heads_ * head_dim_;
    full_out_dim_ = num_q_heads_ * head_dim_;
    attn_output_gate_ = args.attn_output_gate;
    layer_types_ = args.layer_types;
}

bool Qwen35Model::detect_weight_prefix(ModelWeights* sf) {
    static const char* prefixes[] = {
        "model.language_model.",
        "model.",
        "",
    };
    for (auto* p : prefixes) {
        char name[256];
        snprintf(name, sizeof(name), "%sembed_tokens.weight", p);
        if (sf->find(name)) {
            weight_prefix_ = p;
            LOG("Weight prefix: \"%s\"\n", weight_prefix_.c_str());
            return true;
        }
    }
    fprintf(stderr, "Cannot detect weight prefix: embed_tokens.weight not found\n");
    return false;
}

bool Qwen35Model::load(const std::string& model_dir, const std::string& backend,
                       const std::string& coreml_dir_override) {
    // 1. Read config.json and parse args
    std::string config_path = model_dir + "/config.json";
    std::ifstream f(config_path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open %s\n", config_path.c_str());
        return false;
    }
    json j = json::parse(f);
    Qwen35Args args = Qwen35Args::from_json(j);
    apply_args(args);

    // 2. Open model weights (single-file or sharded)
    auto sf = ModelWeights::open(model_dir);
    if (!sf) {
        fprintf(stderr, "Failed to open model weights in %s\n", model_dir.c_str());
        return false;
    }

    // 3. Detect weight naming convention
    if (!detect_weight_prefix(sf.get())) {
        return false;
    }

    // Infer dims from safetensors
    char embed_name[256], gate_name[256];
    snprintf(embed_name, sizeof(embed_name), "%sembed_tokens.weight", weight_prefix_.c_str());
    snprintf(gate_name, sizeof(gate_name), "%slayers.0.mlp.gate_proj.weight", weight_prefix_.c_str());

    const SFTensor* embed = sf->find(embed_name);
    if (!embed || embed->ndims != 2) {
        fprintf(stderr, "Cannot infer dims: missing or invalid embed_tokens.weight\n");
        return false;
    }
    const SFTensor* gate = sf->find(gate_name);
    if (!gate || gate->ndims != 2) {
        fprintf(stderr, "Cannot infer dims: missing or invalid gate_proj.weight\n");
        return false;
    }

    hidden_size_ = (int)embed->shape[1];
    vocab_size_ = (int)embed->shape[0];
    intermediate_size_ = (int)gate->shape[0];

    LOG("Model dims: hidden=%d intermediate=%d vocab=%d layers=%d\n",
        hidden_size_, intermediate_size_, vocab_size_, num_layers_);

    // 3. Init ANE
    ane_init();

    // Allocate scratch buffers
    x_ = (float*)calloc(hidden_size_, sizeof(float));
    x_norm_ = (float*)calloc(hidden_size_, sizeof(float));
    logits_ = (float*)calloc(vocab_size_, sizeof(float));
    // Extra space for norm-fused DeltaNet: in_proj_a + in_proj_b at tail
    int qkv_buf_size = std::max(lin_qkv_dim_ + lin_total_val_ + 2 * lin_num_val_heads_,
                                full_q_dim_ + full_kv_dim_ * 2);
    scratch_qkv_ = (float*)calloc(qkv_buf_size, sizeof(float));
    scratch_conv_ = (float*)calloc(lin_qkv_dim_, sizeof(float));
    scratch_y_ = (float*)calloc(lin_total_val_, sizeof(float));
    scratch_attn_ = (float*)calloc(std::max(full_out_dim_, lin_total_val_), sizeof(float));
    // scratch_tmp_: a_vec (lin_num_val_heads_) + b_vec (lin_num_val_heads_) + silu_tmp (lin_qkv_dim_)
    scratch_tmp_ = (float*)calloc((size_t)lin_num_val_heads_ * 2 + lin_qkv_dim_, sizeof(float));
    rope_cos_ = (float*)calloc((size_t)MAX_SEQ_LEN * (rot_dim_ / 2), sizeof(float));
    rope_sin_ = (float*)calloc((size_t)MAX_SEQ_LEN * (rot_dim_ / 2), sizeof(float));

    // Batch prefill buffers
    {
        int deltanet_proj_out = lin_qkv_dim_ + lin_total_val_ + 2 * lin_num_val_heads_;
        int fullattn_proj_out = full_q_dim_ + full_kv_dim_ * 2;
        int max_proj_dim = std::max(deltanet_proj_out, fullattn_proj_out);
        int max_core_dim = std::max(lin_total_val_, full_out_dim_);

        x_batch_ = (float*)calloc((size_t)MAX_PREFILL_BATCH * hidden_size_, sizeof(float));
        proj_batch_ = (float*)calloc((size_t)MAX_PREFILL_BATCH * max_proj_dim, sizeof(float));
        core_batch_ = (float*)calloc((size_t)MAX_PREFILL_BATCH * max_core_dim, sizeof(float));
    }

    // Precompute RoPE trig table
    if (rope_cos_ && rope_sin_) {
        int half_rot = rot_dim_ / 2;
        float inv_freq[half_rot];
        for (int j2 = 0, i = 0; i < rot_dim_; i += 2, j2++) {
            inv_freq[j2] = 1.0f / powf(rope_theta_, (float)i / (float)rot_dim_);
        }
        for (int pos = 0; pos < MAX_SEQ_LEN; pos++) {
            float* cos_row = rope_cos_ + (size_t)pos * half_rot;
            float* sin_row = rope_sin_ + (size_t)pos * half_rot;
            for (int j2 = 0; j2 < half_rot; j2++) {
                float angle = pos * inv_freq[j2];
                cos_row[j2] = cosf(angle);
                sin_row[j2] = sinf(angle);
            }
        }
    }

    // Initialize layers
    layers_.resize(num_layers_);
    delta_states_.resize(num_layers_);
    kv_caches_.resize(num_layers_);
    ane_layers_.resize(num_layers_);

    for (int L = 0; L < num_layers_; L++) {
        if (layer_types_[L] == LayerType::FullAttention) {
            auto& kv = kv_caches_[L];
            kv.k_cache = (float*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, sizeof(float));
            kv.v_cache = (float*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, sizeof(float));
            kv.len = 0;
            kv.start = 0;
            kv.capacity = KV_CACHE_CAPACITY;
        }
        if (layer_types_[L] == LayerType::LinearAttention) {
            auto& ds = delta_states_[L];
            ds.ssm_state = (float*)calloc((size_t)lin_num_val_heads_ * lin_key_dim_ * lin_val_dim_, sizeof(float));
            ds.conv_state = (float*)calloc((size_t)lin_qkv_dim_ * (conv_kernel_ - 1), sizeof(float));
            ds.conv_pos = 0;
        }
    }

    // 4. Load weights + compile ANE kernels
    if (!load_weights(sf.get())) { return false; }
    // Try loading MTP weights (optional, non-fatal if missing)
    load_mtp_weights(sf.get());
    // Detect pre-converted ANE blob directory
    std::string blob_dir = model_dir + "/ane_weights";
    struct stat st_blob;
    bool has_blobs = (stat(blob_dir.c_str(), &st_blob) == 0 && S_ISDIR(st_blob.st_mode));
    if (has_blobs) {
        LOG("Using pre-converted ANE blobs from %s\n", blob_dir.c_str());
    }

    // Check for CoreML backend
    if (backend == "coreml") {
        std::string coreml_dir = coreml_dir_override.empty()
            ? model_dir + "-coreml" : coreml_dir_override;
        struct stat st_coreml;
        if (stat(coreml_dir.c_str(), &st_coreml) != 0 || !S_ISDIR(st_coreml.st_mode)) {
            fprintf(stderr, "CoreML backend requested but %s not found.\n"
                    "Run: python scripts/export_coreml_model.py --model %s --output %s\n",
                    coreml_dir.c_str(), model_dir.c_str(), coreml_dir.c_str());
            return false;
        }
        if (!compile_coreml(coreml_dir)) { return false; }
    } else {
        if (!compile_ane(sf.get(), has_blobs ? blob_dir : "")) { return false; }
    }

    return true;
}

bool Qwen35Model::load_weights(ModelWeights* sf) {
    char name[256];
    const char* pfx = weight_prefix_.c_str();

    snprintf(name, sizeof(name), "%sembed_tokens.weight", pfx);
    embed_tokens_ = sf->load_bf16_to_f32(name, (int64_t)vocab_size_ * hidden_size_);
    if (!embed_tokens_) return false;

    snprintf(name, sizeof(name), "%snorm.weight", pfx);
    final_norm_ = sf->load_norm_weight(name, hidden_size_);
    if (!final_norm_) return false;

    for (int L = 0; L < num_layers_; L++) {
        auto& lw = layers_[L];
        lw.type = layer_types_[L];

        snprintf(name, sizeof(name), "%slayers.%d.input_layernorm.weight", pfx, L);
        lw.input_layernorm = sf->load_norm_weight(name, hidden_size_);
        if (!lw.input_layernorm) return false;

        snprintf(name, sizeof(name), "%slayers.%d.post_attention_layernorm.weight", pfx, L);
        lw.post_attention_layernorm = sf->load_norm_weight(name, hidden_size_);
        if (!lw.post_attention_layernorm) return false;

        if (lw.type == LayerType::LinearAttention) {
            auto& dw = lw.deltanet;

            // Note: in_proj_a/b, A_log, dt_bias use linear_num_value_heads (not key_heads)
            snprintf(name, sizeof(name), "%slayers.%d.linear_attn.in_proj_a.weight", pfx, L);
            dw.in_proj_a = sf->load_bf16_to_f32(name, (int64_t)lin_num_val_heads_ * hidden_size_);

            snprintf(name, sizeof(name), "%slayers.%d.linear_attn.in_proj_b.weight", pfx, L);
            dw.in_proj_b = sf->load_bf16_to_f32(name, (int64_t)lin_num_val_heads_ * hidden_size_);

            snprintf(name, sizeof(name), "%slayers.%d.linear_attn.conv1d.weight", pfx, L);
            dw.conv1d_w = sf->load_bf16_to_f32(name, (int64_t)lin_qkv_dim_ * conv_kernel_);

            snprintf(name, sizeof(name), "%slayers.%d.linear_attn.A_log", pfx, L);
            dw.A = sf->load_f32_direct(name, lin_num_val_heads_);
            if (dw.A) {
                for (int i = 0; i < lin_num_val_heads_; i++) dw.A[i] = expf(dw.A[i]);
            }

            snprintf(name, sizeof(name), "%slayers.%d.linear_attn.dt_bias", pfx, L);
            dw.dt_bias = sf->load_bf16_to_f32(name, lin_num_val_heads_);

            snprintf(name, sizeof(name), "%slayers.%d.linear_attn.norm.weight", pfx, L);
            dw.norm_w = sf->load_f32_direct(name, lin_val_dim_);

            if (!dw.in_proj_a || !dw.in_proj_b || !dw.conv1d_w ||
                !dw.A || !dw.dt_bias || !dw.norm_w) {
                fprintf(stderr, "Failed to load DeltaNet weights for layer %d\n", L);
                return false;
            }
        } else {
            auto& fw = lw.full_attn;

            snprintf(name, sizeof(name), "%slayers.%d.self_attn.q_norm.weight", pfx, L);
            fw.q_norm = sf->load_norm_weight(name, head_dim_);

            snprintf(name, sizeof(name), "%slayers.%d.self_attn.k_norm.weight", pfx, L);
            fw.k_norm = sf->load_norm_weight(name, head_dim_);

            if (!fw.q_norm || !fw.k_norm) {
                fprintf(stderr, "Failed to load FullAttn weights for layer %d\n", L);
                return false;
            }
        }
    }

    LOG("All weights loaded successfully\n");
    return true;
}

bool Qwen35Model::load_mtp_weights(ModelWeights* sf) {
    char name[256];

    snprintf(name, sizeof(name), "mtp.pre_fc_norm_hidden.weight");
    mtp_.pre_fc_norm_hidden = sf->load_norm_weight(name, hidden_size_);
    if (!mtp_.pre_fc_norm_hidden) { LOG("MTP weights not found, skipping\n"); return false; }

    snprintf(name, sizeof(name), "mtp.pre_fc_norm_embedding.weight");
    mtp_.pre_fc_norm_embedding = sf->load_norm_weight(name, hidden_size_);

    snprintf(name, sizeof(name), "mtp.fc.weight");
    mtp_.fc = sf->load_bf16_to_f32(name, (int64_t)hidden_size_ * 2 * hidden_size_);

    snprintf(name, sizeof(name), "mtp.layers.0.self_attn.q_proj.weight");
    mtp_.q_proj = sf->load_bf16_to_f32(name, (int64_t)full_q_dim_ * hidden_size_);

    snprintf(name, sizeof(name), "mtp.layers.0.self_attn.k_proj.weight");
    mtp_.k_proj = sf->load_bf16_to_f32(name, (int64_t)full_kv_dim_ * hidden_size_);

    snprintf(name, sizeof(name), "mtp.layers.0.self_attn.v_proj.weight");
    mtp_.v_proj = sf->load_bf16_to_f32(name, (int64_t)full_kv_dim_ * hidden_size_);

    snprintf(name, sizeof(name), "mtp.layers.0.self_attn.o_proj.weight");
    mtp_.o_proj = sf->load_bf16_to_f32(name, (int64_t)hidden_size_ * full_out_dim_);

    snprintf(name, sizeof(name), "mtp.layers.0.self_attn.q_norm.weight");
    mtp_.q_norm = sf->load_norm_weight(name, head_dim_);

    snprintf(name, sizeof(name), "mtp.layers.0.self_attn.k_norm.weight");
    mtp_.k_norm = sf->load_norm_weight(name, head_dim_);

    snprintf(name, sizeof(name), "mtp.layers.0.input_layernorm.weight");
    mtp_.input_layernorm = sf->load_norm_weight(name, hidden_size_);

    snprintf(name, sizeof(name), "mtp.layers.0.post_attention_layernorm.weight");
    mtp_.post_attn_layernorm = sf->load_norm_weight(name, hidden_size_);

    snprintf(name, sizeof(name), "mtp.layers.0.mlp.gate_proj.weight");
    mtp_.gate_proj = sf->load_bf16_to_f32(name, (int64_t)intermediate_size_ * hidden_size_);

    snprintf(name, sizeof(name), "mtp.layers.0.mlp.up_proj.weight");
    mtp_.up_proj = sf->load_bf16_to_f32(name, (int64_t)intermediate_size_ * hidden_size_);

    snprintf(name, sizeof(name), "mtp.layers.0.mlp.down_proj.weight");
    mtp_.down_proj = sf->load_bf16_to_f32(name, (int64_t)hidden_size_ * intermediate_size_);

    snprintf(name, sizeof(name), "mtp.norm.weight");
    mtp_.norm = sf->load_norm_weight(name, hidden_size_);

    if (!mtp_.pre_fc_norm_embedding || !mtp_.fc || !mtp_.q_proj || !mtp_.k_proj ||
        !mtp_.v_proj || !mtp_.o_proj || !mtp_.q_norm || !mtp_.k_norm ||
        !mtp_.input_layernorm || !mtp_.post_attn_layernorm ||
        !mtp_.gate_proj || !mtp_.up_proj || !mtp_.down_proj || !mtp_.norm) {
        fprintf(stderr, "Failed to load some MTP weights\n");
        return false;
    }

    // Allocate MTP KV cache
    mtp_kv_cache_.k_cache = (float*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, sizeof(float));
    mtp_kv_cache_.v_cache = (float*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, sizeof(float));
    mtp_kv_cache_.len = 0;
    mtp_kv_cache_.start = 0;
    mtp_kv_cache_.capacity = KV_CACHE_CAPACITY;

    // Allocate buffer for pre-final hidden state
    pre_final_x_ = (float*)calloc(hidden_size_, sizeof(float));

    mtp_loaded_ = true;
    LOG("MTP weights loaded (1 transformer layer + FC)\n");
    return true;
}

// Convert tensor name to blob path: "a.b.c" → "<dir>/a/b/c.bin"
static std::string blob_path(const std::string& dir, const char* tensor_name) {
    std::string p = dir + "/";
    for (const char* c = tensor_name; *c; c++) {
        p += (*c == '.') ? '/' : *c;
    }
    p += ".bin";
    return p;
}

bool Qwen35Model::compile_ane(ModelWeights* sf, const std::string& blob_dir) {
    if (!ane_available()) {
        fprintf(stderr, "ANE not available, cannot run\n");
        return false;
    }

    bool use_blobs = !blob_dir.empty();
    LOG("Compiling ANE kernels%s...\n", use_blobs ? " (from blobs)" : "");
    char name[256], name2[256], name3[256];
    const char* pfx = weight_prefix_.c_str();

    for (int L = 0; L < num_layers_; L++) {
        LOG("  Layer %d/%d (%s)...\r", L+1, num_layers_,
            layer_types_[L] == LayerType::LinearAttention ? "deltanet" : "full_attn");

        if (layer_types_[L] == LayerType::LinearAttention) {
            snprintf(name, sizeof(name), "%slayers.%d.linear_attn.in_proj_qkv.weight", pfx, L);
            snprintf(name2, sizeof(name2), "%slayers.%d.linear_attn.in_proj_z.weight", pfx, L);

            if (use_blobs) {
                ane_layers_[L].first_proj = ane_compile_fused_2_blob(
                    blob_path(blob_dir, name), lin_qkv_dim_,
                    blob_path(blob_dir, name2), lin_total_val_, hidden_size_);
            } else {
                ane_layers_[L].first_proj = ane_compile_fused_2(
                    sf->get_bf16_ptr(name), lin_qkv_dim_,
                    sf->get_bf16_ptr(name2), lin_total_val_, hidden_size_);
            }
        } else {
            snprintf(name, sizeof(name), "%slayers.%d.self_attn.q_proj.weight", pfx, L);
            snprintf(name2, sizeof(name2), "%slayers.%d.self_attn.k_proj.weight", pfx, L);
            snprintf(name3, sizeof(name3), "%slayers.%d.self_attn.v_proj.weight", pfx, L);

            if (use_blobs) {
                ane_layers_[L].first_proj = ane_compile_fused_3_blob(
                    blob_path(blob_dir, name), full_q_dim_,
                    blob_path(blob_dir, name2), full_kv_dim_,
                    blob_path(blob_dir, name3), full_kv_dim_, hidden_size_);
            } else {
                ane_layers_[L].first_proj = ane_compile_fused_3(
                    sf->get_bf16_ptr(name), full_q_dim_,
                    sf->get_bf16_ptr(name2), full_kv_dim_,
                    sf->get_bf16_ptr(name3), full_kv_dim_, hidden_size_);
            }
        }

        if (!ane_layers_[L].first_proj) {
            fprintf(stderr, "ANE first_proj compile failed for layer %d\n", L);
            return false;
        }

        // O projection
        int attn_dim;
        if (layer_types_[L] == LayerType::LinearAttention) {
            snprintf(name, sizeof(name), "%slayers.%d.linear_attn.out_proj.weight", pfx, L);
            attn_dim = lin_total_val_;
        } else {
            snprintf(name, sizeof(name), "%slayers.%d.self_attn.o_proj.weight", pfx, L);
            attn_dim = full_out_dim_;
        }
        if (use_blobs) {
            ane_layers_[L].o_proj = ane_compile_matmul_blob(blob_path(blob_dir, name), hidden_size_, attn_dim);
        } else {
            ane_layers_[L].o_proj = ane_compile_matmul(sf->get_bf16_ptr(name), hidden_size_, attn_dim);
        }
        if (!ane_layers_[L].o_proj) {
            fprintf(stderr, "ANE o_proj compile failed for layer %d\n", L);
            return false;
        }

        // Fused FFN
        snprintf(name, sizeof(name), "%slayers.%d.mlp.gate_proj.weight", pfx, L);
        snprintf(name2, sizeof(name2), "%slayers.%d.mlp.up_proj.weight", pfx, L);
        snprintf(name3, sizeof(name3), "%slayers.%d.mlp.down_proj.weight", pfx, L);

        if (use_blobs) {
            ane_layers_[L].fused_ffn = ane_compile_fused_ffn_blob(
                blob_path(blob_dir, name), blob_path(blob_dir, name2),
                blob_path(blob_dir, name3), hidden_size_, intermediate_size_);
        } else {
            ane_layers_[L].fused_ffn = ane_compile_fused_ffn(
                sf->get_bf16_ptr(name), sf->get_bf16_ptr(name2),
                sf->get_bf16_ptr(name3), hidden_size_, intermediate_size_);
        }
        if (!ane_layers_[L].fused_ffn) {
            fprintf(stderr, "ANE fused_ffn compile failed for layer %d\n", L);
            return false;
        }
    }

    int compiled = ane_compile_count();
    int cached = ane_cache_loads();
    LOG("  %d ANE layer kernels ready (compiled=%d, cached=%d)\n",
        compiled + cached, compiled, cached);

    // Compile LM head
    if (!compile_lm_head_ane(sf, blob_dir)) {
        LOG("ANE LM head disabled, falling back to CPU\n");
    } else {
        LOG("  LM head ANE enabled (%d chunks)\n", (int)lm_head_kernels_.size());
    }

    return true;
}

bool Qwen35Model::compile_lm_head_ane(ModelWeights* sf, const std::string& blob_dir) {
    bool use_blobs = !blob_dir.empty();

    // For blob mode, we need the embed blob; for bf16 mode, the bf16 pointer
    char embed_name[256];
    snprintf(embed_name, sizeof(embed_name), "%sembed_tokens.weight", weight_prefix_.c_str());

    const uint16_t* embed_bf16 = nullptr;
    if (!use_blobs) {
        embed_bf16 = sf->get_bf16_ptr(embed_name);
        if (!embed_bf16) {
            fprintf(stderr, "ANE LM head: missing embed_tokens BF16 weights\n");
            return false;
        }
    }

    int chunk = lm_head_chunk_;
    if (chunk > vocab_size_) chunk = vocab_size_;

    int chunks = (vocab_size_ + chunk - 1) / chunk;
    lm_head_kernels_.resize(chunks, nullptr);

    LOG("  LM head ANE: compiling %d chunks (chunk=%d)\n", chunks, chunk);
    for (int c = 0; c < chunks; c++) {
        int offset = c * chunk;
        int rows = vocab_size_ - offset;
        if (rows > chunk) rows = chunk;

        LOG("    LM head chunk %d/%d...\r", c + 1, chunks);

        if (use_blobs) {
            // LM head reuses embed_tokens weight, chunked by row offset
            // Blob was written as one file; we need per-chunk blobs or fall back to bf16
            // For now: fall back to bf16 for LM head since embed_tokens is one big blob
            embed_bf16 = sf->get_bf16_ptr(embed_name);
            if (!embed_bf16) return false;
            const uint16_t* chunk_w = embed_bf16 + (int64_t)offset * hidden_size_;
            lm_head_kernels_[c] = ane_compile_matmul(chunk_w, rows, hidden_size_);
        } else {
            const uint16_t* chunk_w = embed_bf16 + (int64_t)offset * hidden_size_;
            lm_head_kernels_[c] = ane_compile_matmul(chunk_w, rows, hidden_size_);
        }
        if (!lm_head_kernels_[c]) {
            fprintf(stderr, "\nANE LM head: compile failed at chunk %d/%d\n", c + 1, chunks);
            free_lm_head_ane();
            return false;
        }
    }
    LOG("    LM head chunk %d/%d done          \n", chunks, chunks);
    ane_lm_head_enabled_ = true;
    lm_head_chunk_ = chunk;
    return true;
}

void Qwen35Model::free_lm_head_ane() {
    for (auto* k : lm_head_kernels_) ane_free(k);
    lm_head_kernels_.clear();
    ane_lm_head_enabled_ = false;
}

bool Qwen35Model::compile_coreml(const std::string& coreml_dir) {
    if (!coreml_available()) {
        fprintf(stderr, "CoreML not available\n");
        return false;
    }

    LOG("Loading CoreML models from %s...\n", coreml_dir.c_str());
    coreml_layers_.resize(num_layers_);
    use_coreml_ = true;

    for (int L = 0; L < num_layers_; L++) {
        LOG("  Layer %d/%d (%s)...\r", L+1, num_layers_,
            layer_types_[L] == LayerType::LinearAttention ? "deltanet" : "full_attn");

        int first_proj_out;
        if (layer_types_[L] == LayerType::LinearAttention) {
            first_proj_out = lin_qkv_dim_ + lin_total_val_;
        } else {
            first_proj_out = full_q_dim_ + full_kv_dim_ * 2;
        }

        // Check for norm-fused first_proj (includes RMSNorm + in_proj_a/b for DeltaNet)
        std::string nfp_path = coreml_dir + "/layer_" + std::to_string(L) + "_norm_first_proj.mlpackage";
        struct stat nfp_st;
        if (stat(nfp_path.c_str(), &nfp_st) == 0) {
            int nf_out = first_proj_out;
            if (layer_types_[L] == LayerType::LinearAttention)
                nf_out += 2 * lin_num_val_heads_;  // in_proj_a + in_proj_b fused
            coreml_layers_[L].first_proj = coreml_load(nfp_path, hidden_size_, nf_out);
            coreml_layers_[L].norm_fused = true;
        } else {
            // Legacy first_proj (no norm)
            std::string fp_path = coreml_dir + "/layer_" + std::to_string(L) + "_first_proj.mlpackage";
            coreml_layers_[L].first_proj = coreml_load(fp_path, hidden_size_, first_proj_out);
        }
        if (!coreml_layers_[L].first_proj) {
            fprintf(stderr, "CoreML first_proj load failed for layer %d\n", L);
            return false;
        }

        // o_proj (same for both modes)
        int attn_dim = (layer_types_[L] == LayerType::LinearAttention) ? lin_total_val_ : full_out_dim_;
        std::string o_proj_path = coreml_dir + "/layer_" + std::to_string(L) + "_o_proj.mlpackage";
        coreml_layers_[L].o_proj = coreml_load(o_proj_path, attn_dim, hidden_size_);
        if (!coreml_layers_[L].o_proj) {
            fprintf(stderr, "CoreML o_proj load failed for layer %d\n", L);
            return false;
        }

        // Check for norm-fused FFN
        std::string nff_path = coreml_dir + "/layer_" + std::to_string(L) + "_norm_fused_ffn.mlpackage";
        struct stat nff_st;
        if (stat(nff_path.c_str(), &nff_st) == 0) {
            coreml_layers_[L].norm_fused_ffn = coreml_load(nff_path, hidden_size_, hidden_size_);
            if (!coreml_layers_[L].norm_fused_ffn) {
                fprintf(stderr, "CoreML norm_fused_ffn load failed for layer %d\n", L);
                return false;
            }
        } else {
            // Legacy fused_ffn (no norm)
            std::string ffn_path = coreml_dir + "/layer_" + std::to_string(L) + "_fused_ffn.mlpackage";
            coreml_layers_[L].fused_ffn = coreml_load(ffn_path, hidden_size_, hidden_size_);
            if (!coreml_layers_[L].fused_ffn) {
                fprintf(stderr, "CoreML fused_ffn load failed for layer %d\n", L);
                return false;
            }
        }
    }
    norm_fused_mode_ = coreml_layers_[0].norm_fused;
    const char* mode = norm_fused_mode_ ? "norm-fused" : "legacy";
    LOG("  %d CoreML layer models loaded (%s)\n", num_layers_ * 3, mode);

    // Initialize batch support for prefill
    if (norm_fused_mode_) {
        bool batch_ok = true;
        for (int L = 0; L < num_layers_ && batch_ok; L++) {
            batch_ok &= coreml_init_batch(coreml_layers_[L].first_proj, MAX_PREFILL_BATCH);
            batch_ok &= coreml_init_batch(coreml_layers_[L].o_proj, MAX_PREFILL_BATCH);
            if (coreml_layers_[L].norm_fused_ffn)
                batch_ok &= coreml_init_batch(coreml_layers_[L].norm_fused_ffn, MAX_PREFILL_BATCH);
        }
        if (batch_ok) {
            batch_enabled_ = true;
            LOG("  Batch prefill enabled (max_batch=%d)\n", MAX_PREFILL_BATCH);
        } else {
            LOG("  Batch prefill disabled (init failed, using single-token)\n");
        }
    }

    // LM head chunks
    for (int c = 0; ; c++) {
        std::string lm_path = coreml_dir + "/lm_head_chunk_" + std::to_string(c) + ".mlpackage";
        struct stat st;
        if (stat(lm_path.c_str(), &st) != 0) break;

        int chunk_rows = LM_HEAD_ANE_CHUNK_MAX;
        int remaining = vocab_size_ - c * LM_HEAD_ANE_CHUNK_MAX;
        if (remaining < chunk_rows) chunk_rows = remaining;

        CoreMLKernel* k = coreml_load(lm_path, hidden_size_, chunk_rows);
        if (!k) {
            fprintf(stderr, "CoreML LM head chunk %d load failed\n", c);
            // Non-fatal: fall back to CPU for LM head
            break;
        }
        coreml_lm_head_.push_back(k);
    }
    if (!coreml_lm_head_.empty()) {
        lm_head_chunk_ = LM_HEAD_ANE_CHUNK_MAX;
        ane_lm_head_enabled_ = true;
        LOG("  LM head: %d CoreML chunks\n", (int)coreml_lm_head_.size());
    } else {
        LOG("  LM head: CPU fallback\n");
    }

    return true;
}

bool Qwen35Model::predict_matvec(int layer, int op, float* output, const float* input,
                                  int in_dim, int out_dim) {
    uint64_t t0 = 0;
    if (timing_enabled_ && op == OP_FIRST_PROJ) t0 = mach_absolute_time();

    bool ok;
    if (use_coreml_) {
        CoreMLKernel* k = nullptr;
        switch (op) {
            case OP_FIRST_PROJ: k = coreml_layers_[layer].first_proj; break;
            case OP_O_PROJ:     k = coreml_layers_[layer].o_proj;     break;
            case OP_FUSED_FFN:  k = coreml_layers_[layer].fused_ffn;  break;
        }
        ok = coreml_predict(k, output, input, in_dim, out_dim);
    } else {
        ANEKernel* k = nullptr;
        switch (op) {
            case OP_FIRST_PROJ: k = ane_layers_[layer].first_proj; break;
            case OP_O_PROJ:     k = ane_layers_[layer].o_proj;     break;
            case OP_FUSED_FFN:  k = ane_layers_[layer].fused_ffn;  break;
        }
        ok = ane_matvec(k, output, input, in_dim, out_dim);
    }

    if (timing_enabled_ && op == OP_FIRST_PROJ) {
        timings_.first_proj_us += to_us(mach_absolute_time() - t0);
    }
    return ok;
}

bool Qwen35Model::forward_deltanet_from_proj(int L, float* proj_output, float* pre_oproj, bool norm_fused) {
    auto& dw = layers_[L].deltanet;
    auto& st = delta_states_[L];

    float* mixed_qkv = proj_output;
    float* z = proj_output + lin_qkv_dim_;

    float* a_vec, *b_vec;
    if (norm_fused) {
        // a_vec/b_vec are at the tail of first_proj output
        a_vec = proj_output + lin_qkv_dim_ + lin_total_val_;
        b_vec = a_vec + lin_num_val_heads_;
    } else {
        // Legacy: not supported in from_proj path
        fprintf(stderr, "forward_deltanet_from_proj requires norm_fused mode\n");
        return false;
    }

    // Causal conv1d + SiLU
    float* conv_out = scratch_conv_;
    conv1d_update(conv_out, st.conv_state, &st.conv_pos, mixed_qkv, dw.conv1d_w, lin_qkv_dim_, conv_kernel_);
    silu_vec_inplace(conv_out, lin_qkv_dim_, scratch_tmp_ + lin_num_val_heads_ * 2);

    // Split into Q, K, V
    float* Q = conv_out;
    float* K = conv_out + lin_total_key_;
    float* V = conv_out + lin_total_key_ * 2;

    // Per-head SSM (GQA-style)
    float* y = scratch_y_;
    float q_scale = 1.0f / sqrtf((float)lin_key_dim_);
    int kv_group = lin_num_val_heads_ / lin_num_key_heads_;

    for (int kh = 0; kh < lin_num_key_heads_; kh++) {
        l2_normalize(Q + kh * lin_key_dim_, lin_key_dim_);
        l2_normalize(K + kh * lin_key_dim_, lin_key_dim_);
        vDSP_vsmul(Q + kh * lin_key_dim_, 1, &q_scale,
                   Q + kh * lin_key_dim_, 1, (vDSP_Length)lin_key_dim_);
    }

    for (int h = 0; h < lin_num_val_heads_; h++) {
        int kh_idx = (kv_group > 1) ? (h / kv_group) : h;
        float* qh = Q + kh_idx * lin_key_dim_;
        float* kh_ptr = K + kh_idx * lin_key_dim_;
        float* vh = V + h * lin_val_dim_;
        float* yh = y + h * lin_val_dim_;
        float* state = st.ssm_state + h * lin_key_dim_ * lin_val_dim_;

        float beta = sigmoid_f(b_vec[h]);
        float decay = expf(-dw.A[h] * softplus_f(a_vec[h] + dw.dt_bias[h]));
        ssm_step(yh, state, qh, kh_ptr, vh, decay, beta, lin_key_dim_, lin_val_dim_);
    }

    // RMSNorm gated
    for (int h = 0; h < lin_num_val_heads_; h++) {
        rmsnorm_gated(pre_oproj + h * lin_val_dim_,
                      y + h * lin_val_dim_,
                      z + h * lin_val_dim_,
                      dw.norm_w, lin_val_dim_);
    }
    return true;
}

bool Qwen35Model::forward_deltanet_core(int L, float* x, float* pre_oproj, bool norm_fused) {
    int first_proj_out = lin_qkv_dim_ + lin_total_val_;
    if (norm_fused) first_proj_out += 2 * lin_num_val_heads_;

    float* qkv_z = scratch_qkv_;
    if (!predict_matvec(L, OP_FIRST_PROJ, qkv_z, x,
                        hidden_size_, first_proj_out)) {
        fprintf(stderr, "first_proj eval failed at layer %d (DeltaNet)\n", L);
        return false;
    }

    if (norm_fused) {
        return forward_deltanet_from_proj(L, qkv_z, pre_oproj, true);
    }

    // Legacy path: compute a/b on CPU
    auto& dw = layers_[L].deltanet;
    float* a_vec = scratch_tmp_;
    float* b_vec = scratch_tmp_ + lin_num_val_heads_;
    matvec(a_vec, dw.in_proj_a, x, lin_num_val_heads_, hidden_size_);
    matvec(b_vec, dw.in_proj_b, x, lin_num_val_heads_, hidden_size_);

    // Copy a/b to tail of qkv_z so from_proj can read them
    // Temporarily enable norm_fused layout by appending a/b
    float* tail_a = qkv_z + lin_qkv_dim_ + lin_total_val_;
    float* tail_b = tail_a + lin_num_val_heads_;
    memcpy(tail_a, a_vec, lin_num_val_heads_ * sizeof(float));
    memcpy(tail_b, b_vec, lin_num_val_heads_ * sizeof(float));

    return forward_deltanet_from_proj(L, qkv_z, pre_oproj, true);
}

bool Qwen35Model::forward_full_attn_from_proj(int L, float* proj_output, float* pre_oproj, int pos) {
    auto& fw = layers_[L].full_attn;
    auto& cache = kv_caches_[L];

    float* q_gate_raw = proj_output;
    float* k_raw = proj_output + full_q_dim_;
    float* v_raw = proj_output + full_q_dim_ + full_kv_dim_;

    // RMSNorm on Q and K per-head
    for (int h = 0; h < num_q_heads_; h++) {
        float* qh = q_gate_raw + (size_t)h * head_dim_ * 2;
        rmsnorm(qh, qh, fw.q_norm, head_dim_, rms_eps_);
    }
    for (int h = 0; h < num_kv_heads_; h++) {
        rmsnorm(k_raw + h * head_dim_, k_raw + h * head_dim_, fw.k_norm, head_dim_, rms_eps_);
    }

    // RoPE
    const float* rope_cos_row = nullptr;
    const float* rope_sin_row = nullptr;
    if (pos >= 0 && pos < MAX_SEQ_LEN && rope_cos_ && rope_sin_) {
        int half_rot = rot_dim_ / 2;
        rope_cos_row = rope_cos_ + (size_t)pos * half_rot;
        rope_sin_row = rope_sin_ + (size_t)pos * half_rot;
    }
    apply_rope_cached(q_gate_raw, k_raw, num_q_heads_, num_kv_heads_,
                      head_dim_, head_dim_ * 2, head_dim_, rot_dim_, pos, rope_theta_,
                      rope_cos_row, rope_sin_row);

    // KV cache update
    int slot;
    if (cache.len < cache.capacity) {
        slot = cache.start + cache.len;
        if (slot >= cache.capacity) slot -= cache.capacity;
        cache.len++;
    } else {
        slot = cache.start;
        cache.start++;
        if (cache.start >= cache.capacity) cache.start = 0;
    }
    size_t kv_stride = (size_t)num_kv_heads_ * head_dim_;
    memcpy(cache.k_cache + (size_t)slot * kv_stride, k_raw, kv_stride * sizeof(float));
    memcpy(cache.v_cache + (size_t)slot * kv_stride, v_raw, kv_stride * sizeof(float));

    // GQA attention
    gqa_attention(pre_oproj, q_gate_raw, cache.k_cache, cache.v_cache,
                  num_q_heads_, num_kv_heads_, head_dim_, head_dim_ * 2,
                  cache.start, cache.len, cache.capacity);

    // Output gate
    if (attn_output_gate_) {
        for (int h = 0; h < num_q_heads_; h++) {
            float* oh = pre_oproj + h * head_dim_;
            const float* gh = q_gate_raw + (size_t)h * head_dim_ * 2 + head_dim_;
            mul_sigmoid_inplace(oh, gh, head_dim_, scratch_tmp_);
        }
    }
    return true;
}

bool Qwen35Model::forward_full_attn_core(int L, float* x, float* pre_oproj, int pos) {
    float* qkv_buf = scratch_qkv_;
    if (!predict_matvec(L, OP_FIRST_PROJ, qkv_buf, x,
                        hidden_size_, full_q_dim_ + full_kv_dim_ * 2)) {
        fprintf(stderr, "first_proj eval failed at layer %d (FullAttn)\n", L);
        return false;
    }
    return forward_full_attn_from_proj(L, qkv_buf, pre_oproj, pos);
}

float* Qwen35Model::forward_mtp(int draft_token, int pos) {
    if (!mtp_loaded_) return nullptr;

    // pre_final_x_ contains the hidden state from the last forward() call
    // (saved before final_norm was applied)

    // 1. Normalize hidden state and embedding
    float* h_norm = scratch_tmp_;
    float* e_norm = scratch_tmp_ + hidden_size_;
    rmsnorm(h_norm, pre_final_x_, mtp_.pre_fc_norm_hidden, hidden_size_, rms_eps_);
    float* embed = embed_tokens_ + (int64_t)draft_token * hidden_size_;
    rmsnorm(e_norm, embed, mtp_.pre_fc_norm_embedding, hidden_size_, rms_eps_);

    // 2. Concatenate [e_norm, h_norm] and FC projection (embedding first, hidden second)
    float combined[hidden_size_ * 2];
    memcpy(combined, e_norm, hidden_size_ * sizeof(float));
    memcpy(combined + hidden_size_, h_norm, hidden_size_ * sizeof(float));

    float* mtp_x = x_norm_;  // reuse scratch buffer
    matvec(mtp_x, mtp_.fc, combined, hidden_size_, hidden_size_ * 2);

    // 3. Transformer layer: attention
    // Pre-attention norm
    float* attn_input = scratch_conv_;  // reuse buffer
    rmsnorm(attn_input, mtp_x, mtp_.input_layernorm, hidden_size_, rms_eps_);

    // QKV projections (CPU matvec)
    float* qkv_buf = scratch_qkv_;
    float* q_raw = qkv_buf;
    float* k_raw = qkv_buf + full_q_dim_;
    float* v_raw = qkv_buf + full_q_dim_ + full_kv_dim_;
    matvec(q_raw, mtp_.q_proj, attn_input, full_q_dim_, hidden_size_);
    matvec(k_raw, mtp_.k_proj, attn_input, full_kv_dim_, hidden_size_);
    matvec(v_raw, mtp_.v_proj, attn_input, full_kv_dim_, hidden_size_);

    // QK norm per-head
    for (int h = 0; h < num_q_heads_; h++) {
        float* qh = q_raw + (size_t)h * head_dim_ * 2;
        rmsnorm(qh, qh, mtp_.q_norm, head_dim_, rms_eps_);
    }
    for (int h = 0; h < num_kv_heads_; h++) {
        rmsnorm(k_raw + h * head_dim_, k_raw + h * head_dim_, mtp_.k_norm, head_dim_, rms_eps_);
    }

    // RoPE
    const float* rope_cos_row = nullptr;
    const float* rope_sin_row = nullptr;
    if (pos >= 0 && pos < MAX_SEQ_LEN && rope_cos_ && rope_sin_) {
        int half_rot = rot_dim_ / 2;
        rope_cos_row = rope_cos_ + (size_t)pos * half_rot;
        rope_sin_row = rope_sin_ + (size_t)pos * half_rot;
    }
    apply_rope_cached(q_raw, k_raw, num_q_heads_, num_kv_heads_,
                      head_dim_, head_dim_ * 2, head_dim_, rot_dim_, pos, rope_theta_,
                      rope_cos_row, rope_sin_row);

    // MTP KV cache update
    auto& cache = mtp_kv_cache_;
    int slot;
    if (cache.len < cache.capacity) {
        slot = cache.start + cache.len;
        if (slot >= cache.capacity) slot -= cache.capacity;
        cache.len++;
    } else {
        slot = cache.start;
        cache.start++;
        if (cache.start >= cache.capacity) cache.start = 0;
    }
    size_t kv_stride = (size_t)num_kv_heads_ * head_dim_;
    memcpy(cache.k_cache + (size_t)slot * kv_stride, k_raw, kv_stride * sizeof(float));
    memcpy(cache.v_cache + (size_t)slot * kv_stride, v_raw, kv_stride * sizeof(float));

    // GQA attention
    float* attn_out = scratch_attn_;
    gqa_attention(attn_out, q_raw, cache.k_cache, cache.v_cache,
                  num_q_heads_, num_kv_heads_, head_dim_, head_dim_ * 2,
                  cache.start, cache.len, cache.capacity);

    // Output gate
    if (attn_output_gate_) {
        for (int h = 0; h < num_q_heads_; h++) {
            float* oh = attn_out + h * head_dim_;
            const float* gh = q_raw + (size_t)h * head_dim_ * 2 + head_dim_;
            mul_sigmoid_inplace(oh, gh, head_dim_, scratch_tmp_);
        }
    }

    // O projection
    float* o_out = scratch_conv_;  // reuse
    matvec(o_out, mtp_.o_proj, attn_out, hidden_size_, full_out_dim_);

    // Residual 1
    for (int i = 0; i < hidden_size_; i++) mtp_x[i] += o_out[i];

    // 4. FFN
    float* ffn_input = scratch_conv_;  // reuse
    rmsnorm(ffn_input, mtp_x, mtp_.post_attn_layernorm, hidden_size_, rms_eps_);

    // Gate + Up -> SiLU -> Down (CPU)
    float* gate_out = scratch_y_;  // reuse, size >= intermediate_size
    float* up_out = scratch_attn_;  // reuse
    // Note: these buffers need to be large enough for intermediate_size
    // scratch_y_ is lin_total_val_ which is 2048 for 0.8B — intermediate is 3584, too small!
    // Use x_batch_ as scratch (MAX_PREFILL_BATCH * hidden_size = 8*1024 = 8192 >= 3584)
    gate_out = x_batch_;
    up_out = x_batch_ + intermediate_size_;
    matvec(gate_out, mtp_.gate_proj, ffn_input, intermediate_size_, hidden_size_);
    matvec(up_out, mtp_.up_proj, ffn_input, intermediate_size_, hidden_size_);

    // SiLU(gate) * up
    for (int i = 0; i < intermediate_size_; i++) {
        float g = gate_out[i];
        gate_out[i] = (g / (1.0f + expf(-g))) * up_out[i];
    }

    float* ffn_out = scratch_conv_;  // reuse
    matvec(ffn_out, mtp_.down_proj, gate_out, hidden_size_, intermediate_size_);

    // Residual 2
    for (int i = 0; i < hidden_size_; i++) mtp_x[i] += ffn_out[i];

    // 5. Final norm
    rmsnorm(mtp_x, mtp_x, mtp_.norm, hidden_size_, rms_eps_);

    // 6. LM head (shared with main model, reuse logits_)
    if (use_coreml_ && !coreml_lm_head_.empty()) {
        int chunks = (int)coreml_lm_head_.size();
        for (int c = 0; c < chunks; c++) {
            int offset = c * lm_head_chunk_;
            int rows = vocab_size_ - offset;
            if (rows > lm_head_chunk_) rows = lm_head_chunk_;
            if (!coreml_predict(coreml_lm_head_[c], logits_ + offset, mtp_x, hidden_size_, rows))
                return nullptr;
        }
    } else if (!use_coreml_ && ane_lm_head_enabled_ && !lm_head_kernels_.empty()) {
        int chunks = (int)lm_head_kernels_.size();
        for (int c = 0; c < chunks; c++) {
            int offset = c * lm_head_chunk_;
            int rows = vocab_size_ - offset;
            if (rows > lm_head_chunk_) rows = lm_head_chunk_;
            if (!ane_matvec(lm_head_kernels_[c], logits_ + offset, mtp_x, hidden_size_, rows))
                return nullptr;
        }
    } else {
        matvec(logits_, embed_tokens_, mtp_x, vocab_size_, hidden_size_);
    }

    return logits_;
}

float* Qwen35Model::forward_batch(const int* tokens, int n_tokens, int start_pos) {
    if (!batch_enabled_ || !norm_fused_mode_ || n_tokens < 1 || n_tokens > MAX_PREFILL_BATCH) {
        // Fallback to single-token loop
        float* logits = nullptr;
        for (int i = 0; i < n_tokens; i++) {
            bool last = (i == n_tokens - 1);
            logits = forward(tokens[i], start_pos + i, last);
            if (!logits) return nullptr;
        }
        return logits;
    }

    // Embedding lookup for all tokens
    for (int t = 0; t < n_tokens; t++)
        memcpy(x_batch_ + (size_t)t * hidden_size_,
               embed_tokens_ + (int64_t)tokens[t] * hidden_size_,
               hidden_size_ * sizeof(float));

    // Compute max projection dims
    int deltanet_proj_out = lin_qkv_dim_ + lin_total_val_ + 2 * lin_num_val_heads_;
    int fullattn_proj_out = full_q_dim_ + full_kv_dim_ * 2;

    for (int L = 0; L < num_layers_; L++) {
        int proj_dim, core_dim;
        if (layer_types_[L] == LayerType::LinearAttention) {
            proj_dim = deltanet_proj_out;
            core_dim = lin_total_val_;
        } else {
            proj_dim = fullattn_proj_out;
            core_dim = full_out_dim_;
        }

        // Batch ANE: first_proj for all tokens
        if (!coreml_predict_batch(coreml_layers_[L].first_proj,
                                   proj_batch_, x_batch_,
                                   hidden_size_, proj_dim, n_tokens)) {
            fprintf(stderr, "Batch first_proj failed at layer %d\n", L);
            return nullptr;
        }

        // Sequential: process each token's attention/SSM core
        for (int t = 0; t < n_tokens; t++) {
            float* proj_t = proj_batch_ + (size_t)t * proj_dim;
            float* core_t = core_batch_ + (size_t)t * core_dim;
            int pos = start_pos + t;

            if (layer_types_[L] == LayerType::LinearAttention) {
                if (!forward_deltanet_from_proj(L, proj_t, core_t, true))
                    return nullptr;
            } else {
                if (!forward_full_attn_from_proj(L, proj_t, core_t, pos))
                    return nullptr;
            }
        }

        // Batch ANE: o_proj for all tokens
        int attn_dim = (layer_types_[L] == LayerType::LinearAttention) ? lin_total_val_ : full_out_dim_;
        if (!coreml_predict_batch(coreml_layers_[L].o_proj,
                                   proj_batch_, core_batch_,
                                   attn_dim, hidden_size_, n_tokens)) {
            fprintf(stderr, "Batch o_proj failed at layer %d\n", L);
            return nullptr;
        }

        // Residual 1 for all tokens
        for (int t = 0; t < n_tokens; t++)
            for (int i = 0; i < hidden_size_; i++)
                x_batch_[t * hidden_size_ + i] += proj_batch_[t * hidden_size_ + i];

        // Batch ANE: norm_fused_ffn for all tokens
        CoreMLKernel* ffn_k = coreml_layers_[L].norm_fused_ffn;
        if (!coreml_predict_batch(ffn_k,
                                   proj_batch_, x_batch_,
                                   hidden_size_, hidden_size_, n_tokens)) {
            fprintf(stderr, "Batch FFN failed at layer %d\n", L);
            return nullptr;
        }

        // Residual 2 for all tokens
        for (int t = 0; t < n_tokens; t++)
            for (int i = 0; i < hidden_size_; i++)
                x_batch_[t * hidden_size_ + i] += proj_batch_[t * hidden_size_ + i];
    }

    // Copy last token hidden state to x_
    memcpy(x_, x_batch_ + (size_t)(n_tokens - 1) * hidden_size_,
           hidden_size_ * sizeof(float));

    // Final norm
    rmsnorm(x_, x_, final_norm_, hidden_size_, rms_eps_);

    // LM head (last token only)
    if (use_coreml_ && !coreml_lm_head_.empty()) {
        int chunks = (int)coreml_lm_head_.size();
        for (int c = 0; c < chunks; c++) {
            int offset = c * lm_head_chunk_;
            int rows = vocab_size_ - offset;
            if (rows > lm_head_chunk_) rows = lm_head_chunk_;
            if (!coreml_predict(coreml_lm_head_[c], logits_ + offset, x_, hidden_size_, rows)) {
                fprintf(stderr, "LM head failed at chunk %d\n", c);
                return nullptr;
            }
        }
    } else {
        matvec(logits_, embed_tokens_, x_, vocab_size_, hidden_size_);
    }

    return logits_;
}

float* Qwen35Model::forward(int token, int pos) {
    return forward(token, pos, true);
}

float* Qwen35Model::forward(int token, int pos, bool compute_logits) {
    uint64_t t0 = 0, t1 = 0;
    bool do_timing = timing_enabled_ && pos > 0;  // skip prompt tokens for cleaner averages

    // Embedding lookup
    memcpy(x_, embed_tokens_ + (int64_t)token * hidden_size_, hidden_size_ * sizeof(float));

    float* pre_oproj = scratch_attn_;

    for (int L = 0; L < num_layers_; L++) {
        bool nf = use_coreml_ && coreml_layers_.size() > 0 && coreml_layers_[L].norm_fused;

        // Pre-attention norm (skip if norm is fused into first_proj)
        if (do_timing) t0 = mach_absolute_time();
        if (!nf) {
            rmsnorm(x_norm_, x_, layers_[L].input_layernorm, hidden_size_, rms_eps_);
        }
        float* proj_input = nf ? x_ : x_norm_;

        // Attention core (first_proj ANE call happens inside)
        if (layer_types_[L] == LayerType::LinearAttention) {
            if (!forward_deltanet_core(L, proj_input, pre_oproj, nf)) return nullptr;
        } else {
            if (!forward_full_attn_core(L, proj_input, pre_oproj, pos)) return nullptr;
        }
        if (do_timing) {
            t1 = mach_absolute_time();
            timings_.cpu_us += to_us(t1 - t0);
        }

        int attn_dim = (layer_types_[L] == LayerType::LinearAttention) ? lin_total_val_ : full_out_dim_;

        // o_proj
        float* attn_out = x_norm_;
        if (do_timing) t0 = mach_absolute_time();
        if (!predict_matvec(L, OP_O_PROJ, attn_out, pre_oproj, attn_dim, hidden_size_)) {
            fprintf(stderr, "o_proj eval failed at layer %d\n", L);
            return nullptr;
        }
        if (do_timing) {
            t1 = mach_absolute_time();
            timings_.o_proj_us += to_us(t1 - t0);
        }

        // Residual 1
        for (int i = 0; i < hidden_size_; i++) x_[i] += attn_out[i];

        // FFN (norm-fused or legacy)
        float* mlp_out = scratch_attn_;
        if (nf && coreml_layers_[L].norm_fused_ffn) {
            // Norm-fused: ANE does norm + FFN, input is raw x
            if (do_timing) t0 = mach_absolute_time();
            if (!coreml_predict(coreml_layers_[L].norm_fused_ffn,
                                 mlp_out, x_, hidden_size_, hidden_size_)) {
                fprintf(stderr, "norm_fused_ffn eval failed at layer %d\n", L);
                return nullptr;
            }
            if (do_timing) {
                t1 = mach_absolute_time();
                timings_.ffn_us += to_us(t1 - t0);
            }
        } else {
            // Legacy: CPU norm + ANE FFN
            rmsnorm(x_norm_, x_, layers_[L].post_attention_layernorm, hidden_size_, rms_eps_);
            if (do_timing) t0 = mach_absolute_time();
            if (!predict_matvec(L, OP_FUSED_FFN, mlp_out, x_norm_, hidden_size_, hidden_size_)) {
                fprintf(stderr, "fused_ffn eval failed at layer %d\n", L);
                return nullptr;
            }
            if (do_timing) {
                t1 = mach_absolute_time();
                timings_.ffn_us += to_us(t1 - t0);
            }
        }

        // Residual 2
        for (int i = 0; i < hidden_size_; i++) x_[i] += mlp_out[i];

    }

    // Save hidden state before final norm (needed by MTP)
    if (mtp_loaded_ && pre_final_x_)
        memcpy(pre_final_x_, x_, hidden_size_ * sizeof(float));

    // Final norm
    rmsnorm(x_, x_, final_norm_, hidden_size_, rms_eps_);

    // Skip LM head during prefill (logits not needed)
    if (!compute_logits) return x_;

    // LM head
    if (do_timing) t0 = mach_absolute_time();
    if (use_coreml_ && !coreml_lm_head_.empty()) {
        bool ok = true;
        int chunks = (int)coreml_lm_head_.size();
        for (int c = 0; c < chunks; c++) {
            int offset = c * lm_head_chunk_;
            int rows = vocab_size_ - offset;
            if (rows > lm_head_chunk_) rows = lm_head_chunk_;
            if (!coreml_predict(coreml_lm_head_[c], logits_ + offset, x_, hidden_size_, rows)) {
                fprintf(stderr, "CoreML LM head eval failed at chunk %d/%d, falling back to CPU\n", c + 1, chunks);
                ok = false;
                break;
            }
        }
        if (!ok) {
            for (auto* k : coreml_lm_head_) coreml_free(k);
            coreml_lm_head_.clear();
            ane_lm_head_enabled_ = false;
            matvec(logits_, embed_tokens_, x_, vocab_size_, hidden_size_);
        }
    } else if (!use_coreml_ && ane_lm_head_enabled_ && !lm_head_kernels_.empty()) {
        bool ok = true;
        int chunks = (int)lm_head_kernels_.size();
        for (int c = 0; c < chunks; c++) {
            int offset = c * lm_head_chunk_;
            int rows = vocab_size_ - offset;
            if (rows > lm_head_chunk_) rows = lm_head_chunk_;
            if (!ane_matvec(lm_head_kernels_[c], logits_ + offset, x_, hidden_size_, rows)) {
                fprintf(stderr, "ANE LM head eval failed at chunk %d/%d, falling back to CPU\n", c + 1, chunks);
                ok = false;
                break;
            }
        }
        if (!ok) {
            free_lm_head_ane();
            matvec(logits_, embed_tokens_, x_, vocab_size_, hidden_size_);
        }
    } else {
        matvec(logits_, embed_tokens_, x_, vocab_size_, hidden_size_);
    }
    if (do_timing) {
        t1 = mach_absolute_time();
        timings_.lm_head_us += to_us(t1 - t0);
    }

    if (do_timing) timings_.count++;

    return logits_;
}

void Qwen35Model::print_timing() {
    if (timings_.count == 0) return;
    int n = timings_.count;

    const char* mode = norm_fused_mode_ ? "norm-fused" : "legacy";
    fprintf(stderr, "\n--- Per-token timing (avg over %d gen tokens, %d layers, %s) ---\n",
            n, num_layers_, mode);

    // cpu_us includes first_proj time (called inside forward_*_core)
    // Subtract first_proj to get pure CPU time
    double first_proj_avg = timings_.first_proj_us / n;
    double cpu_raw_avg = timings_.cpu_us / n;
    double cpu_pure_avg = cpu_raw_avg - first_proj_avg;
    double o_proj_avg = timings_.o_proj_us / n;
    double ffn_avg = timings_.ffn_us / n;
    double lm_head_avg = timings_.lm_head_us / n;
    double total = first_proj_avg + cpu_pure_avg + o_proj_avg + ffn_avg + lm_head_avg;

    const char* fp_label = norm_fused_mode_ ? "norm_first_proj" : "first_proj";
    const char* ffn_label = norm_fused_mode_ ? "norm_ffn" : "fused_ffn";

    fprintf(stderr, "  %-15s %7.2f ms/token  (%.2f ms/call x %d)\n",
            fp_label, first_proj_avg / 1000.0, first_proj_avg / 1000.0 / num_layers_, num_layers_);
    fprintf(stderr, "  cpu (attn):   %7.2f ms/token\n", cpu_pure_avg / 1000.0);
    fprintf(stderr, "  o_proj:       %7.2f ms/token  (%.2f ms/call x %d)\n",
            o_proj_avg / 1000.0, o_proj_avg / 1000.0 / num_layers_, num_layers_);
    fprintf(stderr, "  %-15s %7.2f ms/token  (%.2f ms/call x %d)\n",
            ffn_label, ffn_avg / 1000.0, ffn_avg / 1000.0 / num_layers_, num_layers_);
    fprintf(stderr, "  lm_head:      %7.2f ms/token\n", lm_head_avg / 1000.0);
    fprintf(stderr, "  total:        %7.2f ms/token -> %.1f tok/s\n",
            total / 1000.0, 1000000.0 / total);
}

} // namespace ane_lm
