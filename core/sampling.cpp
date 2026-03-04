#include "sampling.h"
#include "cpu_ops.h"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <algorithm>
#include <unordered_map>

namespace ane_lm {

int sample_token(const float* logits, int vocab_size,
                 const SamplingParams& params,
                 const std::vector<int>& recent_tokens) {
    auto adjusted_buf = std::unique_ptr<float[]>(new float[vocab_size]);
    float* adjusted = adjusted_buf.get();
    memcpy(adjusted, logits, vocab_size * sizeof(float));

    if (!recent_tokens.empty()) {
        int start = std::max(0, (int)recent_tokens.size() - params.repetition_context_size);

        // Count token frequencies in the context window
        std::unordered_map<int, int> freq;
        for (int j = start; j < (int)recent_tokens.size(); j++) {
            int tok = recent_tokens[j];
            if (tok >= 0 && tok < vocab_size) {
                freq[tok]++;
            }
        }

        // Apply repetition penalty + frequency penalty
        for (auto& [tok, count] : freq) {
            // Repetition penalty (https://arxiv.org/abs/1909.05858)
            if (params.repetition_penalty > 1.0f) {
                if (adjusted[tok] > 0.0f) {
                    adjusted[tok] /= params.repetition_penalty;
                } else {
                    adjusted[tok] *= params.repetition_penalty;
                }
            }
            // Frequency penalty: subtract penalty * count from logit
            if (params.frequency_penalty > 0.0f) {
                adjusted[tok] -= params.frequency_penalty * count;
            }
        }
    }

    if (params.temperature <= 0.0f) {
        int max_i = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (adjusted[i] > adjusted[max_i]) max_i = i;
        }
        return max_i;
    }

    float inv_t = 1.0f / params.temperature;
    for (int i = 0; i < vocab_size; i++) adjusted[i] *= inv_t;
    softmax(adjusted, vocab_size);

    float r = (float)drand48();
    float cum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cum += adjusted[i];
        if (cum >= r) { return i; }
    }
    return vocab_size - 1;
}

} // namespace ane_lm
