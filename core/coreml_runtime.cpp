// coreml_runtime.cpp — CoreML model loading and prediction via ObjC runtime
// Provides coreml_predict() as drop-in replacement for ane_matvec() with quantized models
#include <objc/objc.h>
#include <objc/runtime.h>
#include <objc/message.h>
#include <cstdio>
#include <cstring>
#include <string>
#include "coreml_runtime.h"
#include <ane_lm/common.h>

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
    id inputArray;      // MLMultiArray* (pre-allocated, reused)
    id outputFeature;   // Cached output feature name (NSString*)
    id inputFeature;    // Cached input feature name (NSString*)
    int in_dim;
    int out_dim;
};

// ============ MLMultiArray helpers ============

// Create an MLMultiArray with shape [1, dim]
static id create_multi_array(int dim, long dataType = kMLDataTypeFloat32) {
    id shape = ((id(*)(Class,SEL,unsigned long))objc_msgSend)(
        cls("NSMutableArray"), sel("arrayWithCapacity:"), (unsigned long)2);
    ((void(*)(id,SEL,id))objc_msgSend)(shape, sel("addObject:"), ns_number_int(1));
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

// ============ Public API ============

bool coreml_available() {
    return cls("MLModel") != nil;
}

CoreMLKernel* coreml_load(const std::string& model_path, int in_dim, int out_dim,
                           int compute_unit) {
    void* pool = objc_autoreleasePoolPush();

    // Create NSURL from path
    id path_str = ns_str(model_path.c_str());
    id url = ((id(*)(Class,SEL,id))objc_msgSend)(cls("NSURL"), sel("fileURLWithPath:"), path_str);

    // Compile .mlpackage to .mlmodelc if needed
    // [MLModel compileModelAtURL:error:] returns URL to compiled model in temp dir
    id compiled_url = url;
    if (model_path.size() > 10 &&
        model_path.substr(model_path.size() - 10) == ".mlpackage") {
        id err = nil;
        compiled_url = ((id(*)(Class,SEL,id,id*))objc_msgSend)(
            cls("MLModel"), sel("compileModelAtURL:error:"), url, &err);
        if (err || !compiled_url) {
            const char* desc = err ?
                ((const char*(*)(id,SEL))objc_msgSend)(
                    ((id(*)(id,SEL))objc_msgSend)(err, sel("localizedDescription")),
                    sel("UTF8String")) : "nil";
            fprintf(stderr, "[CoreML] Compilation failed for %s: %s\n",
                    model_path.c_str(), desc);
            objc_autoreleasePoolPop(pool);
            return nullptr;
        }
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

    // Pre-allocate input MLMultiArray
    id inputArray = create_multi_array(in_dim);
    if (!inputArray) {
        ((void(*)(id,SEL))objc_msgSend)(model, sel("release"));
        objc_autoreleasePoolPop(pool);
        return nullptr;
    }
    ((id(*)(id,SEL))objc_msgSend)(inputArray, sel("retain"));

    // Retain feature names
    ((id(*)(id,SEL))objc_msgSend)(inputName, sel("retain"));
    ((id(*)(id,SEL))objc_msgSend)(outputName, sel("retain"));

    // Create kernel
    CoreMLKernel* k = new CoreMLKernel();
    k->model = model;
    k->inputArray = inputArray;
    k->inputFeature = inputName;
    k->outputFeature = outputName;
    k->in_dim = in_dim;
    k->out_dim = out_dim;

    objc_autoreleasePoolPop(pool);
    return k;
}

bool coreml_predict(CoreMLKernel* k, float* output, const float* input,
                     int in_dim, int out_dim) {
    void* pool = objc_autoreleasePoolPush();

    // Copy input data to pre-allocated MLMultiArray (FP32)
    float* in_ptr = (float*)multi_array_data_ptr(k->inputArray);
    memcpy(in_ptr, input, in_dim * sizeof(float));

    // Create MLDictionaryFeatureProvider with input
    id featureValue = ((id(*)(Class,SEL,id))objc_msgSend)(
        cls("MLFeatureValue"), sel("featureValueWithMultiArray:"), k->inputArray);

    // Build NSDictionary {inputName: featureValue}
    id keys[] = { k->inputFeature };
    id values[] = { featureValue };
    id dict = ((id(*)(Class,SEL,id*,id*,unsigned long))objc_msgSend)(
        cls("NSDictionary"), sel("dictionaryWithObjects:forKeys:count:"),
        values, keys, 1UL);

    id err = nil;
    id provider = ((id(*)(id,SEL,id,id*))objc_msgSend)(
        ((id(*)(Class,SEL))objc_msgSend)(cls("MLDictionaryFeatureProvider"), sel("alloc")),
        sel("initWithDictionary:error:"),
        dict, &err);

    if (err || !provider) {
        fprintf(stderr, "[CoreML] Feature provider creation failed\n");
        objc_autoreleasePoolPop(pool);
        return false;
    }

    // Run prediction
    err = nil;
    id result = ((id(*)(id,SEL,id,id*))objc_msgSend)(
        k->model, sel("predictionFromFeatures:error:"),
        provider, &err);

    if (err || !result) {
        const char* desc = err ?
            ((const char*(*)(id,SEL))objc_msgSend)(
                ((id(*)(id,SEL))objc_msgSend)(err, sel("localizedDescription")),
                sel("UTF8String")) : "nil result";
        fprintf(stderr, "[CoreML] Prediction failed: %s\n", desc);
        objc_autoreleasePoolPop(pool);
        return false;
    }

    // Extract output MLMultiArray
    id outFeature = ((id(*)(id,SEL,id))objc_msgSend)(
        result, sel("featureValueForName:"), k->outputFeature);
    id outArray = ((id(*)(id,SEL))objc_msgSend)(outFeature, sel("multiArrayValue"));

    // Copy output — handle FP16 output arrays (CoreML models often use FP16)
    long outType = multi_array_data_type(outArray);
    void* out_ptr = multi_array_data_ptr(outArray);

    if (outType == kMLDataTypeFloat16) {
        // Convert FP16 → FP32
        _Float16* fp16 = (_Float16*)out_ptr;
        for (int i = 0; i < out_dim; i++) {
            output[i] = (float)fp16[i];
        }
    } else {
        // FP32 — direct copy
        memcpy(output, out_ptr, out_dim * sizeof(float));
    }

    objc_autoreleasePoolPop(pool);
    return true;
}

void coreml_free(CoreMLKernel* k) {
    if (!k) return;
    ((void(*)(id,SEL))objc_msgSend)(k->inputArray, sel("release"));
    ((void(*)(id,SEL))objc_msgSend)(k->inputFeature, sel("release"));
    ((void(*)(id,SEL))objc_msgSend)(k->outputFeature, sel("release"));
    ((void(*)(id,SEL))objc_msgSend)(k->model, sel("release"));
    delete k;
}

} // namespace ane_lm
