#pragma once

#include "../../core/model_loader.h"
#include "../../core/ane_runtime.h"
#include "../../core/coreml_runtime.h"
#include <nlohmann/json.hpp>
#include <memory>
#include <string>
#include <vector>

namespace ane_lm {

enum class LayerType {
    LinearAttention,
    FullAttention,
};

// Abstract base class for LLM models
class LLMModel {
public:
    virtual ~LLMModel() = default;
    virtual bool load(const std::string& model_dir, const std::string& backend = "ane") = 0;
    virtual float* forward(int token_id, int pos) = 0;
    virtual void reset() = 0;
    virtual int vocab_size() const = 0;
    virtual void set_timing(bool enabled) { (void)enabled; }
    virtual void print_timing() {}
};

// Model-owned config (mirrors mlx-lm.cpp style)
struct Qwen35Args {
    int hidden_size = 1024;
    int num_hidden_layers = 24;
    int num_attention_heads = 8;
    int num_key_value_heads = 2;
    int head_dim = 256;
    int intermediate_size = 3584;
    int vocab_size = 248320;
    int full_attention_interval = 4;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000000.0f;
    float partial_rotary_factor = 0.25f;
    int linear_num_key_heads = 16;
    int linear_key_head_dim = 128;
    int linear_value_head_dim = 128;
    int linear_num_value_heads = 16;
    int linear_conv_kernel_dim = 4;
    bool tie_word_embeddings = true;
    bool attn_output_gate = true;

    // Derived
    int key_dim() const { return linear_key_head_dim * linear_num_key_heads; }
    int value_dim() const { return linear_value_head_dim * linear_num_value_heads; }
    int conv_dim() const { return 2 * key_dim() + value_dim(); }
    int rotation_dim() const { return static_cast<int>(head_dim * partial_rotary_factor); }

    std::vector<LayerType> layer_types;

    static Qwen35Args from_json(const nlohmann::json& config);
};

// Qwen3.5 model implementation
class Qwen35Model : public LLMModel {
public:
    ~Qwen35Model() override;
    bool load(const std::string& model_dir, const std::string& backend = "ane") override;
    float* forward(int token_id, int pos) override;
    void reset() override;
    int vocab_size() const override { return vocab_size_; }
    void set_timing(bool enabled) override { timing_enabled_ = enabled; }
    void print_timing() override;

private:
    // Config
    int hidden_size_ = 0;
    int intermediate_size_ = 0;
    int vocab_size_ = 0;
    int num_layers_ = 0;
    int num_q_heads_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int rot_dim_ = 0;
    float rope_theta_ = 0;
    float rms_eps_ = 0;
    int lin_num_key_heads_ = 0;
    int lin_num_val_heads_ = 0;
    int lin_key_dim_ = 0;
    int lin_val_dim_ = 0;
    int lin_total_key_ = 0;
    int lin_total_val_ = 0;
    int lin_qkv_dim_ = 0;
    int conv_kernel_ = 0;
    int full_q_dim_ = 0;
    int full_kv_dim_ = 0;
    int full_out_dim_ = 0;
    bool attn_output_gate_ = true;
    std::string weight_prefix_;  // e.g., "model.language_model." or "model." or ""

    static constexpr int MAX_SEQ_LEN = 4096;
    static constexpr int KV_CACHE_CAPACITY = 2048;
    static constexpr int LM_HEAD_ANE_CHUNK_MAX = 16384;

    std::vector<LayerType> layer_types_;

    // DeltaNet weights
    struct DeltaNetWeights {
        float* in_proj_a = nullptr;
        float* in_proj_b = nullptr;
        float* conv1d_w = nullptr;
        float* A = nullptr;
        float* dt_bias = nullptr;
        float* norm_w = nullptr;
    };

    // Full attention weights
    struct FullAttnWeights {
        float* q_norm = nullptr;
        float* k_norm = nullptr;
    };

    // Layer weights
    struct LayerWeights {
        LayerType type;
        DeltaNetWeights deltanet;
        FullAttnWeights full_attn;
        float* input_layernorm = nullptr;
        float* post_attention_layernorm = nullptr;
    };

    // DeltaNet state
    struct DeltaNetState {
        float* ssm_state = nullptr;
        float* conv_state = nullptr;
        int conv_pos = 0;
    };

    // KV cache
    struct KVCache {
        float* k_cache = nullptr;
        float* v_cache = nullptr;
        int len = 0;
        int start = 0;
        int capacity = 0;
    };

    // Model data
    std::vector<LayerWeights> layers_;
    float* embed_tokens_ = nullptr;
    float* final_norm_ = nullptr;

    std::vector<DeltaNetState> delta_states_;
    std::vector<KVCache> kv_caches_;
    std::vector<LayerANEKernels> ane_layers_;

    // CoreML backend
    bool use_coreml_ = false;
    struct LayerCoreMLKernels {
        CoreMLKernel* first_proj = nullptr;
        CoreMLKernel* o_proj = nullptr;       // legacy (non-fused)
        CoreMLKernel* fused_ffn = nullptr;    // legacy (non-fused)
        CoreMLKernel2* post_attn = nullptr;   // fused: o_proj + residual + norm + FFN + residual
    };
    std::vector<LayerCoreMLKernels> coreml_layers_;
    std::vector<CoreMLKernel*> coreml_lm_head_;

    // LM head ANE kernels
    std::vector<ANEKernel*> lm_head_kernels_;
    int lm_head_chunk_ = LM_HEAD_ANE_CHUNK_MAX;
    bool ane_lm_head_enabled_ = false;

    // Scratch buffers
    float* x_ = nullptr;
    float* x_norm_ = nullptr;
    float* logits_ = nullptr;
    float* scratch_qkv_ = nullptr;
    float* scratch_conv_ = nullptr;
    float* scratch_y_ = nullptr;
    float* scratch_attn_ = nullptr;
    float* scratch_tmp_ = nullptr;
    float* rope_cos_ = nullptr;
    float* rope_sin_ = nullptr;

    void apply_args(const Qwen35Args& args);
    bool detect_weight_prefix(ModelWeights* sf);
    bool load_weights(ModelWeights* sf);
    bool compile_ane(ModelWeights* sf, const std::string& blob_dir);
    bool compile_lm_head_ane(ModelWeights* sf, const std::string& blob_dir);
    void free_lm_head_ane();
    bool compile_coreml(const std::string& coreml_dir);

    // Dispatch helper: calls coreml_predict or ane_matvec based on backend
    bool predict_matvec(int layer, int op, float* output, const float* input,
                        int in_dim, int out_dim);
    // op indices for predict_matvec
    static constexpr int OP_FIRST_PROJ = 0;
    static constexpr int OP_O_PROJ = 1;
    static constexpr int OP_FUSED_FFN = 2;

    bool forward_deltanet_core(int L, float* x, float* pre_oproj);
    bool forward_full_attn_core(int L, float* x, float* pre_oproj, int pos);

    // Per-layer timing
    bool timing_enabled_ = false;
    struct ForwardTimings {
        double first_proj_us = 0;
        double post_attn_us = 0;   // fused: o_proj + norm + FFN
        double o_proj_us = 0;      // legacy separate
        double ffn_us = 0;         // legacy separate
        double cpu_us = 0;         // attention core + norms + residuals
        double lm_head_us = 0;
        int count = 0;
    } timings_;
};

} // namespace ane_lm
