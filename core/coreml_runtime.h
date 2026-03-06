#pragma once

#include <string>

namespace ane_lm {

// Opaque CoreML kernel handle (single input)
struct CoreMLKernel;

// Opaque CoreML kernel handle (two inputs)
struct CoreMLKernel2;

// Load a compiled CoreML model (.mlpackage directory) — single input
// compute_unit: 0 = CPU_ONLY, 1 = CPU_AND_GPU, 2 = ALL, 3 = CPU_AND_NE
CoreMLKernel* coreml_load(const std::string& model_path, int in_dim, int out_dim,
                           int compute_unit = 3);

// Run prediction: copy input → predict → copy output
// Same signature as ane_matvec() for drop-in replacement
bool coreml_predict(CoreMLKernel* k, float* output, const float* input,
                     int in_dim, int out_dim);

// Free a CoreML kernel
void coreml_free(CoreMLKernel* k);

// Load a 2-input CoreML model (e.g. fused post_attn)
CoreMLKernel2* coreml_load_2input(const std::string& model_path,
                                    int in1_dim, int in2_dim, int out_dim,
                                    int compute_unit = 3);

// Run prediction with two inputs
bool coreml_predict_2input(CoreMLKernel2* k, float* output,
                            const float* input1, const float* input2,
                            int in1_dim, int in2_dim, int out_dim);

// Free a 2-input CoreML kernel
void coreml_free_2input(CoreMLKernel2* k);

// Check if CoreML is available
bool coreml_available();

} // namespace ane_lm
