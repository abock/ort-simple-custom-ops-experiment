#pragma once

#include <onnxruntime/core/session/onnxruntime_c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OrtSimpleCustomOp {
    struct OrtCustomOp base_op;
    const OrtApi* api;
    const char* name;
    size_t input_count;
    const ONNXTensorElementDataType* input_types;
    size_t output_count;
    const ONNXTensorElementDataType* output_types;
    void (ORT_API_CALL* kernel_compute_callback)(
        _In_ const OrtApi* op_kernel,
        _In_ const OrtKernelContext* context);
} OrtSimpleCustomOp;

ORT_EXPORT void OrtInitializeCustomOpDefaults(_In_ OrtCustomOp* op);

ORT_EXPORT void OrtInitializeSimpleCustomOp(
    _In_ const OrtApi* api,
    _In_ const char* name,
    _In_ size_t input_count,
    _In_ const ONNXTensorElementDataType* input_types,
    _In_ size_t output_count,
    _In_ const ONNXTensorElementDataType* output_types,
    void (ORT_API_CALL* kernel_compute_callback)(
        _In_ const OrtApi* op_kernel,
        _In_ const OrtKernelContext* context),
    _Out_ OrtSimpleCustomOp* op);

#ifdef __cplusplus
}
#endif
