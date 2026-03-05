#pragma once

#include <string>

namespace ane_lm {

// Opaque CoreML kernel handle
struct CoreMLKernel;

// Load a compiled CoreML model (.mlpackage directory)
// compute_unit: 0 = CPU_ONLY, 1 = CPU_AND_GPU, 2 = ALL, 3 = CPU_AND_NE
CoreMLKernel* coreml_load(const std::string& model_path, int in_dim, int out_dim,
                           int compute_unit = 3);

// Run prediction: copy input → predict → copy output
// Same signature as ane_matvec() for drop-in replacement
bool coreml_predict(CoreMLKernel* k, float* output, const float* input,
                     int in_dim, int out_dim);

// Free a CoreML kernel
void coreml_free(CoreMLKernel* k);

// Check if CoreML is available
bool coreml_available();

} // namespace ane_lm
