// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <onnxruntime/core/session/onnxruntime_c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

ORT_EXPORT void OrtInitializeCustomOpDefaults(_In_ OrtCustomOp* op);

typedef struct OrtSimpleCustomOp OrtSimpleCustomOp;

typedef struct OrtSimpleCustomOpIOConfig {
    size_t count;
    ONNXTensorElementDataType homogenous_type;
    ONNXTensorElementDataType* heterogeneous_types;
} OrtSimpleCustomOpIOConfig;

typedef struct OrtSimpleCustomOpConfig {
    char* name;
    OrtSimpleCustomOpIOConfig inputs;
    OrtSimpleCustomOpIOConfig outputs;
    void (ORT_API_CALL* kernel_compute)(
        _In_ const OrtSimpleCustomOp* op,
        _In_ const OrtApi* ort,
        _In_ const OrtKernelContext* context);
} OrtSimpleCustomOpConfig;

ORT_EXPORT OrtStatus* OrtCreateSimpleCustomOp(
    _In_ const OrtApi* ort,
    _In_ OrtAllocator* allocator,
    _In_ const OrtSimpleCustomOpConfig* config,
    _Out_ OrtSimpleCustomOp** op);

typedef struct OrtSimpleCustomOpIO {
    size_t index;
    const OrtValue *value;
    OrtTensorTypeAndShapeInfo *type_and_shape_info;
    void* buffer;
    size_t buffer_len;
    int64_t* dims;
    size_t dims_len;
} OrtSimpleCustomOpIO;

OrtStatus* OrtSimpleCustomOpIORelease(
    _In_ const OrtSimpleCustomOp *op,
    _In_ OrtSimpleCustomOpIO* io);

OrtStatus* OrtSimpleCustomOpGetInput(
    _In_ const OrtSimpleCustomOp* op,
    _In_ const OrtKernelContext* context,
    _In_ size_t index,
    _Out_ OrtSimpleCustomOpIO* input);

OrtStatus* OrtSimpleCustomOpGetOutput(
    _In_ const OrtSimpleCustomOp* op,
    _In_ const OrtKernelContext* context,
    _In_ size_t index,
    _In_ const int64_t* dims,
    _In_ size_t dims_len,
    _Out_ OrtSimpleCustomOpIO* input);

OrtStatus* OrtSimpleCustomOpRegister(
    _In_ const OrtApi* ort,
    _In_ OrtAllocator* allocator,
    _In_ const char* custom_op_domain_name,
    _In_ const OrtSimpleCustomOpConfig* custom_op_configs,
    _In_ size_t custom_op_configs_len,
    _Out_ OrtCustomOpDomain** custom_op_domain,
    _Out_ OrtSimpleCustomOp** custom_ops);

#ifdef __cplusplus
}
#endif
