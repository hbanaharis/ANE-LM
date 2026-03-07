// coreml_runtime.cpp — CoreML model loading and prediction via ObjC runtime
// Provides coreml_predict() as drop-in replacement for ane_matvec() with quantized models
#include <objc/objc.h>
#include <objc/runtime.h>
#include <objc/message.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include "coreml_runtime.h"
#include <ane_lm/common.h>
#include <Accelerate/Accelerate.h>

// MLMultiArrayDataType enum values
static constexpr long kMLDataTypeFloat16 = 65552;  // 0x10010
static constexpr long kMLDataTypeFloat32 = 65568;  // 0x10020
static constexpr long kMLDataTypeFloat64 = 65600;  // 0x10040

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void*);

namespace ane_lm {

// ============ ObjC runtime helpers ============

static inline SEL    sel(const char* n) { return sel_registerName(n); }
static inline Class  cls(const char* n) { return (Class)objc_getClass(n); }

static id ns_str(const char* s) {
    return ((id(*)(Class,SEL,const char*))objc_msgSend)(cls("NSString"), sel("stringWithUTF8String:"), s);
}

// NSNumber helpers for MLMultiArray shape
static id ns_number_int(int v) {
    return ((id(*)(Class,SEL,int))objc_msgSend)(cls("NSNumber"), sel("numberWithInt:"), v);
}

// ============ CoreMLKernel struct ============

struct CoreMLKernel {
    id model;           // MLModel*
    id inputArray;      // MLMultiArray* (pre-allocated, reused) — single-token
    id provider;        // MLDictionaryFeatureProvider* (cached, reused) — single-token
    id outputFeature;   // Cached output feature name (NSString*)
    id inputName;       // Retained input feature name (NSString*)
    int in_dim;
    int out_dim;
    bool input_fp16;    // Whether input MLMultiArray is FP16
    bool output_fp16;   // Whether output MLMultiArray is FP16

    // Batch support (initialized by coreml_init_batch)
    static constexpr int MAX_BATCH_IDX = 4;
    int max_batch = 0;                          // 0 = batch not initialized
    id batchArrays[MAX_BATCH_IDX] = {};         // [1,dim], [2,dim], [4,dim], [8,dim]
    id batchProviders[MAX_BATCH_IDX] = {};      // corresponding providers
};

// ============ MLMultiArray helpers ============

// Create an MLMultiArray with shape [rows, dim]
static id create_multi_array(int dim, long dataType = kMLDataTypeFloat32, int rows = 1) {
    id shape = ((id(*)(Class,SEL,unsigned long))objc_msgSend)(
        cls("NSMutableArray"), sel("arrayWithCapacity:"), (unsigned long)2);
    ((void(*)(id,SEL,id))objc_msgSend)(shape, sel("addObject:"), ns_number_int(rows));
    ((void(*)(id,SEL,id))objc_msgSend)(shape, sel("addObject:"), ns_number_int(dim));

    id err = nil;
    id array = ((id(*)(id,SEL,id,long,id*))objc_msgSend)(
        ((id(*)(Class,SEL))objc_msgSend)(cls("MLMultiArray"), sel("alloc")),
        sel("initWithShape:dataType:error:"),
        shape, dataType, &err);

    if (err) {
        const char* desc = ((const char*(*)(id,SEL))objc_msgSend)(
            ((id(*)(id,SEL))objc_msgSend)(err, sel("localizedDescription")),
            sel("UTF8String"));
        fprintf(stderr, "[CoreML] MLMultiArray creation failed: %s\n", desc);
        return nil;
    }
    return array;
}

// Get the raw data pointer from MLMultiArray
static void* multi_array_data_ptr(id array) {
    return ((void*(*)(id,SEL))objc_msgSend)(array, sel("dataPointer"));
}

// Get the data type of an MLMultiArray
static long multi_array_data_type(id array) {
    return ((long(*)(id,SEL))objc_msgSend)(array, sel("dataType"));
}

// ============ Disk cache helpers ============

// Return path to cached .mlmodelc for a given .mlpackage
// Cache location: same directory, replacing .mlpackage with .mlmodelc
static std::string cached_mlmodelc_path(const std::string& mlpackage_path) {
    // models/Qwen3.5-0.8B-coreml/layer_0_first_proj.mlpackage
    //   → models/Qwen3.5-0.8B-coreml/layer_0_first_proj.mlmodelc
    std::string base = mlpackage_path.substr(0, mlpackage_path.size() - 10); // strip .mlpackage
    return base + ".mlmodelc";
}

// Check if path exists and is a directory
static bool dir_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// Get modification time of a path (file or directory)
static time_t path_mtime(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return 0;
    return st.st_mtime;
}

// Copy directory recursively using NSFileManager
static bool copy_dir(const std::string& src, const std::string& dst) {
    void* pool = objc_autoreleasePoolPush();
    id fm = ((id(*)(Class,SEL))objc_msgSend)(cls("NSFileManager"), sel("defaultManager"));
    id srcStr = ns_str(src.c_str());
    id dstStr = ns_str(dst.c_str());

    id err = nil;
    bool ok = ((bool(*)(id,SEL,id,id,id*))objc_msgSend)(
        fm, sel("copyItemAtPath:toPath:error:"), srcStr, dstStr, &err);

    if (!ok && err) {
        const char* desc = ((const char*(*)(id,SEL))objc_msgSend)(
            ((id(*)(id,SEL))objc_msgSend)(err, sel("localizedDescription")),
            sel("UTF8String"));
        fprintf(stderr, "[CoreML] Failed to cache compiled model: %s\n", desc);
    }
    objc_autoreleasePoolPop(pool);
    return ok;
}

// Get the string path from an NSURL
static std::string nsurl_to_path(id url) {
    id pathStr = ((id(*)(id,SEL))objc_msgSend)(url, sel("path"));
    const char* cstr = ((const char*(*)(id,SEL))objc_msgSend)(pathStr, sel("UTF8String"));
    return std::string(cstr);
}

// ============ Public API ============

bool coreml_available() {
    return cls("MLModel") != nil;
}

CoreMLKernel* coreml_load(const std::string& model_path, int in_dim, int out_dim,
                           int compute_unit) {
    void* pool = objc_autoreleasePoolPush();

    id compiled_url = nil;
    bool is_mlpackage = model_path.size() > 10 &&
        model_path.substr(model_path.size() - 10) == ".mlpackage";

    if (is_mlpackage) {
        // Check for cached .mlmodelc on disk
        std::string cache_path = cached_mlmodelc_path(model_path);
        if (dir_exists(cache_path) && path_mtime(cache_path) >= path_mtime(model_path)) {
            // Use cached compiled model
            compiled_url = ((id(*)(Class,SEL,id))objc_msgSend)(
                cls("NSURL"), sel("fileURLWithPath:"), ns_str(cache_path.c_str()));
        } else {
            // Compile and cache
            id url = ((id(*)(Class,SEL,id))objc_msgSend)(
                cls("NSURL"), sel("fileURLWithPath:"), ns_str(model_path.c_str()));

            id err = nil;
            id temp_url = ((id(*)(Class,SEL,id,id*))objc_msgSend)(
                cls("MLModel"), sel("compileModelAtURL:error:"), url, &err);
            if (err || !temp_url) {
                const char* desc = err ?
                    ((const char*(*)(id,SEL))objc_msgSend)(
                        ((id(*)(id,SEL))objc_msgSend)(err, sel("localizedDescription")),
                        sel("UTF8String")) : "nil";
                fprintf(stderr, "[CoreML] Compilation failed for %s: %s\n",
                        model_path.c_str(), desc);
                objc_autoreleasePoolPop(pool);
                return nullptr;
            }

            // Copy compiled model from temp dir to persistent cache
            std::string temp_path = nsurl_to_path(temp_url);
            // Remove stale cache if exists
            if (dir_exists(cache_path)) {
                id fm = ((id(*)(Class,SEL))objc_msgSend)(cls("NSFileManager"), sel("defaultManager"));
                ((bool(*)(id,SEL,id,id*))objc_msgSend)(
                    fm, sel("removeItemAtPath:error:"), ns_str(cache_path.c_str()), (id*)nullptr);
            }
            if (copy_dir(temp_path, cache_path)) {
                compiled_url = ((id(*)(Class,SEL,id))objc_msgSend)(
                    cls("NSURL"), sel("fileURLWithPath:"), ns_str(cache_path.c_str()));
            } else {
                // Fallback: use temp URL directly
                compiled_url = temp_url;
            }
        }
    } else {
        // Already a .mlmodelc or other compiled format
        compiled_url = ((id(*)(Class,SEL,id))objc_msgSend)(
            cls("NSURL"), sel("fileURLWithPath:"), ns_str(model_path.c_str()));
    }

    // Create MLModelConfiguration
    id config = ((id(*)(id,SEL))objc_msgSend)(
        ((id(*)(Class,SEL))objc_msgSend)(cls("MLModelConfiguration"), sel("alloc")),
        sel("init"));

    // Set compute units: MLComputeUnits enum
    // 0 = cpuOnly, 1 = cpuAndGPU, 2 = all, 3 = cpuAndNeuralEngine
    ((void(*)(id,SEL,long))objc_msgSend)(config, sel("setComputeUnits:"), (long)compute_unit);

    // Load model
    id err = nil;
    id model = ((id(*)(Class,SEL,id,id,id*))objc_msgSend)(
        cls("MLModel"), sel("modelWithContentsOfURL:configuration:error:"),
        compiled_url, config, &err);

    if (err || !model) {
        const char* desc = err ?
            ((const char*(*)(id,SEL))objc_msgSend)(
                ((id(*)(id,SEL))objc_msgSend)(err, sel("localizedDescription")),
                sel("UTF8String")) : "nil model";
        fprintf(stderr, "[CoreML] Failed to load %s: %s\n", model_path.c_str(), desc);
        objc_autoreleasePoolPop(pool);
        return nullptr;
    }

    // Retain the model (prevent autorelease)
    ((id(*)(id,SEL))objc_msgSend)(model, sel("retain"));

    // Get input/output feature names from model description
    id desc = ((id(*)(id,SEL))objc_msgSend)(model, sel("modelDescription"));
    id inputDescs = ((id(*)(id,SEL))objc_msgSend)(desc, sel("inputDescriptionsByName"));
    id outputDescs = ((id(*)(id,SEL))objc_msgSend)(desc, sel("outputDescriptionsByName"));

    id inputKeys = ((id(*)(id,SEL))objc_msgSend)(inputDescs, sel("allKeys"));
    id outputKeys = ((id(*)(id,SEL))objc_msgSend)(outputDescs, sel("allKeys"));

    id inputName = ((id(*)(id,SEL,unsigned long))objc_msgSend)(inputKeys, sel("objectAtIndex:"), 0UL);
    id outputName = ((id(*)(id,SEL,unsigned long))objc_msgSend)(outputKeys, sel("objectAtIndex:"), 0UL);

    // Probe input data type from model description
    id inputDescObj = ((id(*)(id,SEL,id))objc_msgSend)(inputDescs, sel("objectForKey:"), inputName);
    id constraint = ((id(*)(id,SEL))objc_msgSend)(inputDescObj, sel("multiArrayConstraint"));
    long inputDataType = ((long(*)(id,SEL))objc_msgSend)(constraint, sel("dataType"));
    bool input_fp16 = (inputDataType == kMLDataTypeFloat16);

    // Pre-allocate input MLMultiArray (matching model's expected type)
    id inputArray = create_multi_array(in_dim, input_fp16 ? kMLDataTypeFloat16 : kMLDataTypeFloat32);
    if (!inputArray) {
        ((void(*)(id,SEL))objc_msgSend)(model, sel("release"));
        objc_autoreleasePoolPop(pool);
        return nullptr;
    }
    ((id(*)(id,SEL))objc_msgSend)(inputArray, sel("retain"));

    // Build cached feature provider (reused across all predictions)
    // The provider wraps the same MLMultiArray — updating its dataPointer
    // contents before each predict() is sufficient.
    id featureValue = ((id(*)(Class,SEL,id))objc_msgSend)(
        cls("MLFeatureValue"), sel("featureValueWithMultiArray:"), inputArray);
    ((id(*)(id,SEL))objc_msgSend)(featureValue, sel("retain"));

    id fKeys[] = { inputName };
    id fValues[] = { featureValue };
    id dict = ((id(*)(Class,SEL,id*,id*,unsigned long))objc_msgSend)(
        cls("NSDictionary"), sel("dictionaryWithObjects:forKeys:count:"),
        fValues, fKeys, 1UL);

    id providerErr = nil;
    id provider = ((id(*)(id,SEL,id,id*))objc_msgSend)(
        ((id(*)(Class,SEL))objc_msgSend)(cls("MLDictionaryFeatureProvider"), sel("alloc")),
        sel("initWithDictionary:error:"),
        dict, &providerErr);

    if (providerErr || !provider) {
        fprintf(stderr, "[CoreML] Failed to create cached feature provider\n");
        ((void(*)(id,SEL))objc_msgSend)(featureValue, sel("release"));
        ((void(*)(id,SEL))objc_msgSend)(inputArray, sel("release"));
        ((void(*)(id,SEL))objc_msgSend)(model, sel("release"));
        objc_autoreleasePoolPop(pool);
        return nullptr;
    }
    ((id(*)(id,SEL))objc_msgSend)(provider, sel("retain"));

    // Retain output feature name
    ((id(*)(id,SEL))objc_msgSend)(outputName, sel("retain"));

    // Probe output data type with a dummy prediction
    bool output_fp16 = false;
    {
        id probeErr = nil;
        id probeResult = ((id(*)(id,SEL,id,id*))objc_msgSend)(
            model, sel("predictionFromFeatures:error:"),
            provider, &probeErr);
        if (probeResult) {
            id outFeat = ((id(*)(id,SEL,id))objc_msgSend)(
                probeResult, sel("featureValueForName:"), outputName);
            id outArr = ((id(*)(id,SEL))objc_msgSend)(outFeat, sel("multiArrayValue"));
            output_fp16 = (multi_array_data_type(outArr) == kMLDataTypeFloat16);
        }
    }

    // Create kernel
    CoreMLKernel* k = new CoreMLKernel();
    k->model = model;
    k->inputArray = inputArray;
    k->provider = provider;
    k->outputFeature = outputName;
    k->inputName = inputName;
    ((id(*)(id,SEL))objc_msgSend)(inputName, sel("retain"));
    k->in_dim = in_dim;
    k->out_dim = out_dim;
    k->input_fp16 = input_fp16;
    k->output_fp16 = output_fp16;
    k->max_batch = 0;

    objc_autoreleasePoolPop(pool);
    return k;
}

bool coreml_predict(CoreMLKernel* k, float* output, const float* input,
                     int in_dim, int out_dim) {
    void* pool = objc_autoreleasePoolPush();

    // Copy input data to pre-allocated MLMultiArray
    if (k->input_fp16) {
        // Convert FP32 → FP16 via Accelerate
        void* in_ptr = multi_array_data_ptr(k->inputArray);
        vImage_Buffer src = { (void*)input, 1, (unsigned long)in_dim, (unsigned long)(in_dim * sizeof(float)) };
        vImage_Buffer dst = { in_ptr,       1, (unsigned long)in_dim, (unsigned long)(in_dim * sizeof(uint16_t)) };
        vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0);
    } else {
        float* in_ptr = (float*)multi_array_data_ptr(k->inputArray);
        memcpy(in_ptr, input, in_dim * sizeof(float));
    }

    // Run prediction using cached provider
    id err = nil;
    id result = ((id(*)(id,SEL,id,id*))objc_msgSend)(
        k->model, sel("predictionFromFeatures:error:"),
        k->provider, &err);

    if (!result) {
        if (err) {
            const char* desc = ((const char*(*)(id,SEL))objc_msgSend)(
                ((id(*)(id,SEL))objc_msgSend)(err, sel("localizedDescription")),
                sel("UTF8String"));
            fprintf(stderr, "[CoreML] Prediction failed: %s\n", desc);
        }
        objc_autoreleasePoolPop(pool);
        return false;
    }

    // Extract output MLMultiArray
    id outFeature = ((id(*)(id,SEL,id))objc_msgSend)(
        result, sel("featureValueForName:"), k->outputFeature);
    id outArray = ((id(*)(id,SEL))objc_msgSend)(outFeature, sel("multiArrayValue"));
    void* out_ptr = multi_array_data_ptr(outArray);

    if (k->output_fp16) {
        // Vectorized FP16 → FP32 via Accelerate
        vImage_Buffer src = { out_ptr, 1, (unsigned long)out_dim, (unsigned long)(out_dim * 2) };
        vImage_Buffer dst = { output,  1, (unsigned long)out_dim, (unsigned long)(out_dim * 4) };
        vImageConvert_Planar16FtoPlanarF(&src, &dst, 0);
    } else {
        memcpy(output, out_ptr, out_dim * sizeof(float));
    }

    objc_autoreleasePoolPop(pool);
    return true;
}

void coreml_free(CoreMLKernel* k) {
    if (!k) return;
    ((void(*)(id,SEL))objc_msgSend)(k->provider, sel("release"));
    ((void(*)(id,SEL))objc_msgSend)(k->inputArray, sel("release"));
    ((void(*)(id,SEL))objc_msgSend)(k->outputFeature, sel("release"));
    ((void(*)(id,SEL))objc_msgSend)(k->inputName, sel("release"));
    // Free batch arrays/providers
    for (int i = 0; i < CoreMLKernel::MAX_BATCH_IDX; i++) {
        if (k->batchArrays[i])
            ((void(*)(id,SEL))objc_msgSend)(k->batchArrays[i], sel("release"));
        if (k->batchProviders[i])
            ((void(*)(id,SEL))objc_msgSend)(k->batchProviders[i], sel("release"));
    }
    ((void(*)(id,SEL))objc_msgSend)(k->model, sel("release"));
    delete k;
}

// ============ Batch CoreML API ============

static constexpr int BATCH_SIZES[] = {1, 2, 4, 8};

static int batch_index(int batch_size) {
    for (int i = 0; i < CoreMLKernel::MAX_BATCH_IDX; i++) {
        if (BATCH_SIZES[i] == batch_size) return i;
    }
    return -1;
}

bool coreml_init_batch(CoreMLKernel* k, int max_batch) {
    void* pool = objc_autoreleasePoolPush();

    long dtype = k->input_fp16 ? kMLDataTypeFloat16 : kMLDataTypeFloat32;
    k->max_batch = max_batch;

    for (int i = 0; i < CoreMLKernel::MAX_BATCH_IDX; i++) {
        int b = BATCH_SIZES[i];
        if (b > max_batch) break;

        // Create [b, in_dim] MLMultiArray
        id array = create_multi_array(k->in_dim, dtype, b);
        if (!array) {
            fprintf(stderr, "[CoreML] Failed to create batch array size %d\n", b);
            objc_autoreleasePoolPop(pool);
            return false;
        }
        ((id(*)(id,SEL))objc_msgSend)(array, sel("retain"));

        // Wrap in feature provider
        id featureValue = ((id(*)(Class,SEL,id))objc_msgSend)(
            cls("MLFeatureValue"), sel("featureValueWithMultiArray:"), array);
        ((id(*)(id,SEL))objc_msgSend)(featureValue, sel("retain"));

        id fKeys[] = { k->inputName };
        id fValues[] = { featureValue };
        id dict = ((id(*)(Class,SEL,id*,id*,unsigned long))objc_msgSend)(
            cls("NSDictionary"), sel("dictionaryWithObjects:forKeys:count:"),
            fValues, fKeys, 1UL);

        id err = nil;
        id provider = ((id(*)(id,SEL,id,id*))objc_msgSend)(
            ((id(*)(Class,SEL))objc_msgSend)(cls("MLDictionaryFeatureProvider"), sel("alloc")),
            sel("initWithDictionary:error:"),
            dict, &err);
        if (err || !provider) {
            fprintf(stderr, "[CoreML] Failed to create batch provider for size %d\n", b);
            ((void(*)(id,SEL))objc_msgSend)(featureValue, sel("release"));
            ((void(*)(id,SEL))objc_msgSend)(array, sel("release"));
            objc_autoreleasePoolPop(pool);
            return false;
        }
        ((id(*)(id,SEL))objc_msgSend)(provider, sel("retain"));
        ((void(*)(id,SEL))objc_msgSend)(featureValue, sel("release"));

        k->batchArrays[i] = array;
        k->batchProviders[i] = provider;
    }

    // Probe: verify model supports batch shapes by doing a test prediction with batch=2
    if (max_batch >= 2 && k->batchProviders[1]) {
        id probeErr = nil;
        id probeResult = ((id(*)(id,SEL,id,id*))objc_msgSend)(
            k->model, sel("predictionFromFeatures:error:"),
            k->batchProviders[1], &probeErr);
        if (!probeResult) {
            // Model doesn't support batch — clean up and disable
            for (int i = 0; i < CoreMLKernel::MAX_BATCH_IDX; i++) {
                if (k->batchArrays[i]) {
                    ((void(*)(id,SEL))objc_msgSend)(k->batchArrays[i], sel("release"));
                    k->batchArrays[i] = nil;
                }
                if (k->batchProviders[i]) {
                    ((void(*)(id,SEL))objc_msgSend)(k->batchProviders[i], sel("release"));
                    k->batchProviders[i] = nil;
                }
            }
            k->max_batch = 0;
            objc_autoreleasePoolPop(pool);
            return false;
        }
    }

    objc_autoreleasePoolPop(pool);
    return true;
}

bool coreml_predict_batch(CoreMLKernel* k, float* output, const float* input,
                           int in_dim, int out_dim, int batch_size) {
    int idx = batch_index(batch_size);
    if (idx < 0 || !k->batchArrays[idx]) {
        fprintf(stderr, "[CoreML] Batch size %d not initialized\n", batch_size);
        return false;
    }

    void* pool = objc_autoreleasePoolPush();

    int total_in = batch_size * in_dim;
    int total_out = batch_size * out_dim;

    // Copy input data to batch MLMultiArray
    if (k->input_fp16) {
        void* in_ptr = multi_array_data_ptr(k->batchArrays[idx]);
        vImage_Buffer src = { (void*)input, 1, (unsigned long)total_in, (unsigned long)(total_in * sizeof(float)) };
        vImage_Buffer dst = { in_ptr,       1, (unsigned long)total_in, (unsigned long)(total_in * sizeof(uint16_t)) };
        vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0);
    } else {
        float* in_ptr = (float*)multi_array_data_ptr(k->batchArrays[idx]);
        memcpy(in_ptr, input, total_in * sizeof(float));
    }

    // Run prediction
    id err = nil;
    id result = ((id(*)(id,SEL,id,id*))objc_msgSend)(
        k->model, sel("predictionFromFeatures:error:"),
        k->batchProviders[idx], &err);

    if (!result) {
        if (err) {
            const char* desc = ((const char*(*)(id,SEL))objc_msgSend)(
                ((id(*)(id,SEL))objc_msgSend)(err, sel("localizedDescription")),
                sel("UTF8String"));
            fprintf(stderr, "[CoreML] Batch prediction failed (batch=%d): %s\n", batch_size, desc);
        }
        objc_autoreleasePoolPop(pool);
        return false;
    }

    // Extract output
    id outFeature = ((id(*)(id,SEL,id))objc_msgSend)(
        result, sel("featureValueForName:"), k->outputFeature);
    id outArray = ((id(*)(id,SEL))objc_msgSend)(outFeature, sel("multiArrayValue"));
    void* out_ptr = multi_array_data_ptr(outArray);

    if (k->output_fp16) {
        vImage_Buffer src = { out_ptr, 1, (unsigned long)total_out, (unsigned long)(total_out * 2) };
        vImage_Buffer dst = { output,  1, (unsigned long)total_out, (unsigned long)(total_out * 4) };
        vImageConvert_Planar16FtoPlanarF(&src, &dst, 0);
    } else {
        memcpy(output, out_ptr, total_out * sizeof(float));
    }

    objc_autoreleasePoolPop(pool);
    return true;
}

// ============ 2-Input CoreML API ============

struct CoreMLKernel2 {
    id model;
    id inputArray1;     // MLMultiArray* (pre-allocated, reused)
    id inputArray2;     // MLMultiArray* (pre-allocated, reused)
    id provider;        // MLDictionaryFeatureProvider* (cached, reused)
    id outputFeature;   // Cached output feature name (NSString*)
    int in1_dim;
    int in2_dim;
    int out_dim;
    bool output_fp16;
};

CoreMLKernel2* coreml_load_2input(const std::string& model_path,
                                    int in1_dim, int in2_dim, int out_dim,
                                    int compute_unit) {
    void* pool = objc_autoreleasePoolPush();

    // Compile/load model (same logic as single-input)
    id compiled_url = nil;
    bool is_mlpackage = model_path.size() > 10 &&
        model_path.substr(model_path.size() - 10) == ".mlpackage";

    if (is_mlpackage) {
        std::string cache_path = cached_mlmodelc_path(model_path);
        if (dir_exists(cache_path) && path_mtime(cache_path) >= path_mtime(model_path)) {
            compiled_url = ((id(*)(Class,SEL,id))objc_msgSend)(
                cls("NSURL"), sel("fileURLWithPath:"), ns_str(cache_path.c_str()));
        } else {
            id url = ((id(*)(Class,SEL,id))objc_msgSend)(
                cls("NSURL"), sel("fileURLWithPath:"), ns_str(model_path.c_str()));
            id err = nil;
            id temp_url = ((id(*)(Class,SEL,id,id*))objc_msgSend)(
                cls("MLModel"), sel("compileModelAtURL:error:"), url, &err);
            if (err || !temp_url) {
                const char* desc = err ?
                    ((const char*(*)(id,SEL))objc_msgSend)(
                        ((id(*)(id,SEL))objc_msgSend)(err, sel("localizedDescription")),
                        sel("UTF8String")) : "nil";
                fprintf(stderr, "[CoreML] Compilation failed for %s: %s\n",
                        model_path.c_str(), desc);
                objc_autoreleasePoolPop(pool);
                return nullptr;
            }
            std::string temp_path = nsurl_to_path(temp_url);
            if (dir_exists(cache_path)) {
                id fm = ((id(*)(Class,SEL))objc_msgSend)(cls("NSFileManager"), sel("defaultManager"));
                ((bool(*)(id,SEL,id,id*))objc_msgSend)(
                    fm, sel("removeItemAtPath:error:"), ns_str(cache_path.c_str()), (id*)nullptr);
            }
            if (copy_dir(temp_path, cache_path)) {
                compiled_url = ((id(*)(Class,SEL,id))objc_msgSend)(
                    cls("NSURL"), sel("fileURLWithPath:"), ns_str(cache_path.c_str()));
            } else {
                compiled_url = temp_url;
            }
        }
    } else {
        compiled_url = ((id(*)(Class,SEL,id))objc_msgSend)(
            cls("NSURL"), sel("fileURLWithPath:"), ns_str(model_path.c_str()));
    }

    // Create configuration
    id config = ((id(*)(id,SEL))objc_msgSend)(
        ((id(*)(Class,SEL))objc_msgSend)(cls("MLModelConfiguration"), sel("alloc")),
        sel("init"));
    ((void(*)(id,SEL,long))objc_msgSend)(config, sel("setComputeUnits:"), (long)compute_unit);

    // Load model
    id err = nil;
    id model = ((id(*)(Class,SEL,id,id,id*))objc_msgSend)(
        cls("MLModel"), sel("modelWithContentsOfURL:configuration:error:"),
        compiled_url, config, &err);
    if (err || !model) {
        const char* desc = err ?
            ((const char*(*)(id,SEL))objc_msgSend)(
                ((id(*)(id,SEL))objc_msgSend)(err, sel("localizedDescription")),
                sel("UTF8String")) : "nil model";
        fprintf(stderr, "[CoreML] Failed to load 2-input model %s: %s\n", model_path.c_str(), desc);
        objc_autoreleasePoolPop(pool);
        return nullptr;
    }
    ((id(*)(id,SEL))objc_msgSend)(model, sel("retain"));

    // Get input/output feature names from model description
    id desc = ((id(*)(id,SEL))objc_msgSend)(model, sel("modelDescription"));
    id inputDescs = ((id(*)(id,SEL))objc_msgSend)(desc, sel("inputDescriptionsByName"));
    id outputDescs = ((id(*)(id,SEL))objc_msgSend)(desc, sel("outputDescriptionsByName"));

    id inputKeys = ((id(*)(id,SEL))objc_msgSend)(inputDescs, sel("allKeys"));
    id outputKeys = ((id(*)(id,SEL))objc_msgSend)(outputDescs, sel("allKeys"));

    // Get the two input names (sorted to ensure consistent ordering)
    id sortedInputKeys = ((id(*)(id,SEL,SEL))objc_msgSend)(inputKeys, sel("sortedArrayUsingSelector:"),
        sel("compare:"));
    id inputName1 = ((id(*)(id,SEL,unsigned long))objc_msgSend)(sortedInputKeys, sel("objectAtIndex:"), 0UL);
    id inputName2 = ((id(*)(id,SEL,unsigned long))objc_msgSend)(sortedInputKeys, sel("objectAtIndex:"), 1UL);
    id outputName = ((id(*)(id,SEL,unsigned long))objc_msgSend)(outputKeys, sel("objectAtIndex:"), 0UL);

    // Determine which input is which based on names
    // Our export uses "attn_output" and "x_residual"
    // After sorting: "attn_output" < "x_residual"
    // So sorted[0] = attn_output (in1_dim), sorted[1] = x_residual (in2_dim)
    int dim_for_name1 = in1_dim;  // attn_output
    int dim_for_name2 = in2_dim;  // x_residual

    // Pre-allocate input MLMultiArrays
    id inputArray1 = create_multi_array(dim_for_name1);
    id inputArray2 = create_multi_array(dim_for_name2);
    if (!inputArray1 || !inputArray2) {
        ((void(*)(id,SEL))objc_msgSend)(model, sel("release"));
        objc_autoreleasePoolPop(pool);
        return nullptr;
    }
    ((id(*)(id,SEL))objc_msgSend)(inputArray1, sel("retain"));
    ((id(*)(id,SEL))objc_msgSend)(inputArray2, sel("retain"));

    // Build cached feature provider with two inputs
    id featureValue1 = ((id(*)(Class,SEL,id))objc_msgSend)(
        cls("MLFeatureValue"), sel("featureValueWithMultiArray:"), inputArray1);
    id featureValue2 = ((id(*)(Class,SEL,id))objc_msgSend)(
        cls("MLFeatureValue"), sel("featureValueWithMultiArray:"), inputArray2);
    ((id(*)(id,SEL))objc_msgSend)(featureValue1, sel("retain"));
    ((id(*)(id,SEL))objc_msgSend)(featureValue2, sel("retain"));

    id fKeys[] = { inputName1, inputName2 };
    id fValues[] = { featureValue1, featureValue2 };
    id dict = ((id(*)(Class,SEL,id*,id*,unsigned long))objc_msgSend)(
        cls("NSDictionary"), sel("dictionaryWithObjects:forKeys:count:"),
        fValues, fKeys, 2UL);

    id providerErr = nil;
    id provider = ((id(*)(id,SEL,id,id*))objc_msgSend)(
        ((id(*)(Class,SEL))objc_msgSend)(cls("MLDictionaryFeatureProvider"), sel("alloc")),
        sel("initWithDictionary:error:"),
        dict, &providerErr);

    if (providerErr || !provider) {
        fprintf(stderr, "[CoreML] Failed to create 2-input cached feature provider\n");
        ((void(*)(id,SEL))objc_msgSend)(featureValue1, sel("release"));
        ((void(*)(id,SEL))objc_msgSend)(featureValue2, sel("release"));
        ((void(*)(id,SEL))objc_msgSend)(inputArray1, sel("release"));
        ((void(*)(id,SEL))objc_msgSend)(inputArray2, sel("release"));
        ((void(*)(id,SEL))objc_msgSend)(model, sel("release"));
        objc_autoreleasePoolPop(pool);
        return nullptr;
    }
    ((id(*)(id,SEL))objc_msgSend)(provider, sel("retain"));
    ((id(*)(id,SEL))objc_msgSend)(outputName, sel("retain"));

    // Probe output data type with a dummy prediction
    bool output_fp16 = false;
    {
        id probeErr = nil;
        id probeResult = ((id(*)(id,SEL,id,id*))objc_msgSend)(
            model, sel("predictionFromFeatures:error:"),
            provider, &probeErr);
        if (probeResult) {
            id outFeat = ((id(*)(id,SEL,id))objc_msgSend)(
                probeResult, sel("featureValueForName:"), outputName);
            id outArr = ((id(*)(id,SEL))objc_msgSend)(outFeat, sel("multiArrayValue"));
            output_fp16 = (multi_array_data_type(outArr) == kMLDataTypeFloat16);
        }
    }

    CoreMLKernel2* k = new CoreMLKernel2();
    k->model = model;
    k->inputArray1 = inputArray1;
    k->inputArray2 = inputArray2;
    k->provider = provider;
    k->outputFeature = outputName;
    k->in1_dim = dim_for_name1;
    k->in2_dim = dim_for_name2;
    k->out_dim = out_dim;
    k->output_fp16 = output_fp16;

    objc_autoreleasePoolPop(pool);
    return k;
}

bool coreml_predict_2input(CoreMLKernel2* k, float* output,
                            const float* input1, const float* input2,
                            int in1_dim, int in2_dim, int out_dim) {
    void* pool = objc_autoreleasePoolPush();

    // Copy both inputs to pre-allocated MLMultiArrays
    float* in1_ptr = (float*)multi_array_data_ptr(k->inputArray1);
    memcpy(in1_ptr, input1, in1_dim * sizeof(float));

    float* in2_ptr = (float*)multi_array_data_ptr(k->inputArray2);
    memcpy(in2_ptr, input2, in2_dim * sizeof(float));

    // Run prediction using cached provider
    id err = nil;
    id result = ((id(*)(id,SEL,id,id*))objc_msgSend)(
        k->model, sel("predictionFromFeatures:error:"),
        k->provider, &err);

    if (!result) {
        if (err) {
            const char* desc = ((const char*(*)(id,SEL))objc_msgSend)(
                ((id(*)(id,SEL))objc_msgSend)(err, sel("localizedDescription")),
                sel("UTF8String"));
            fprintf(stderr, "[CoreML] 2-input prediction failed: %s\n", desc);
        }
        objc_autoreleasePoolPop(pool);
        return false;
    }

    // Extract output
    id outFeature = ((id(*)(id,SEL,id))objc_msgSend)(
        result, sel("featureValueForName:"), k->outputFeature);
    id outArray = ((id(*)(id,SEL))objc_msgSend)(outFeature, sel("multiArrayValue"));
    void* out_ptr = multi_array_data_ptr(outArray);

    if (k->output_fp16) {
        vImage_Buffer src = { out_ptr, 1, (unsigned long)out_dim, (unsigned long)(out_dim * 2) };
        vImage_Buffer dst = { output,  1, (unsigned long)out_dim, (unsigned long)(out_dim * 4) };
        vImageConvert_Planar16FtoPlanarF(&src, &dst, 0);
    } else {
        memcpy(output, out_ptr, out_dim * sizeof(float));
    }

    objc_autoreleasePoolPop(pool);
    return true;
}

void coreml_free_2input(CoreMLKernel2* k) {
    if (!k) return;
    ((void(*)(id,SEL))objc_msgSend)(k->provider, sel("release"));
    ((void(*)(id,SEL))objc_msgSend)(k->inputArray1, sel("release"));
    ((void(*)(id,SEL))objc_msgSend)(k->inputArray2, sel("release"));
    ((void(*)(id,SEL))objc_msgSend)(k->outputFeature, sel("release"));
    ((void(*)(id,SEL))objc_msgSend)(k->model, sel("release"));
    delete k;
}

} // namespace ane_lm
