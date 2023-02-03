#include <stdbool.h>

#include "new-custom-op-api.h"

static const char* _Default_ExecutionProviderType(const OrtCustomOp* op)
{
    return NULL;
}

static OrtCustomOpInputOutputCharacteristic _Default_InputCharacteristic(const OrtCustomOp*op, size_t index)
{
    return INPUT_OUTPUT_REQUIRED;
}

static OrtCustomOpInputOutputCharacteristic _Default_OutputCharacteristic(const OrtCustomOp* op, size_t index)
{
    return INPUT_OUTPUT_REQUIRED;
}

static OrtMemType _Default_InputMemoryType(const OrtCustomOp* op, size_t index)
{
    return OrtMemTypeDefault;
}

static int _Default_VariadicInputMinArity(const OrtCustomOp* op)
{
    return 1;
}

static int _Default_VariadicInputHomogeneity(const OrtCustomOp* op)
{
    return true;
}

static int _Default_VariadicOutputMinArity(const OrtCustomOp* op)
{
    return 1;
}

static int _Default_VariadicOutputHomogeneity(const OrtCustomOp* op)
{
    return true;
}

void OrtInitializeCustomOpDefaults(_In_ OrtCustomOp* op)
{
    op->version = ORT_API_VERSION;
    op->GetExecutionProviderType = _Default_ExecutionProviderType;
    op->GetInputCharacteristic = _Default_InputCharacteristic;
    op->GetOutputCharacteristic = _Default_OutputCharacteristic;
    op->GetInputMemoryType = _Default_InputMemoryType;
    op->GetVariadicInputMinArity = _Default_VariadicInputMinArity;
    op->GetVariadicInputHomogeneity = _Default_VariadicInputHomogeneity;
    op->GetVariadicOutputMinArity = _Default_VariadicOutputMinArity;
    op->GetVariadicOutputHomogeneity = _Default_VariadicOutputHomogeneity;
}

static const char* _Simple_GetName(const OrtCustomOp* op)
{
    return ((struct OrtSimpleCustomOp*)op)->name;
}

static size_t _Simple_GetInputTypeCount(const OrtCustomOp* op)
{
    return ((struct OrtSimpleCustomOp*)op)->input_count;
}

static ONNXTensorElementDataType _Simple_GetInputType(const OrtCustomOp* op, size_t index)
{
    return ((struct OrtSimpleCustomOp*)op)->input_types[index];
}

static size_t _Simple_GetOutputTypeCount(const OrtCustomOp* op)
{
    return ((struct OrtSimpleCustomOp*)op)->output_count;
}

static ONNXTensorElementDataType _Simple_GetOutputType(const OrtCustomOp* op, size_t index)
{
    return ((struct OrtSimpleCustomOp*)op)->output_types[index];
}

static void* _Simple_CreateKernel(const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info)
{
    return (void*)op;
}

static void _Simple_KernelCompute(void* kernel, OrtKernelContext* context)
{
    const struct OrtSimpleCustomOp* op = (struct OrtSimpleCustomOp*)kernel;
    op->kernel_compute_callback(op->api, context);
}

static void _Simple_KernelDestroy(void* kernel)
{
}

void OrtInitializeSimpleCustomOp(
    _In_ const OrtApi* api,
    _In_ const char* name,
    _In_ size_t input_count,
    _In_ const ONNXTensorElementDataType* input_types,
    _In_ size_t output_count,
    _In_ const ONNXTensorElementDataType* output_types,
    void (ORT_API_CALL* kernel_compute_callback)(
        _In_ const OrtApi* op_kernel,
        _In_ const OrtKernelContext* context),
    _Out_ OrtSimpleCustomOp* op)
{
    op->api = api;
    op->name = name;
    op->input_count = input_count;
    op->input_types = input_types;
    op->output_count = output_count;
    op->output_types = output_types;
    op->kernel_compute_callback = kernel_compute_callback;

    OrtInitializeCustomOpDefaults(&op->base_op);
    op->base_op.GetName = _Simple_GetName;
    op->base_op.GetInputTypeCount = _Simple_GetInputTypeCount;
    op->base_op.GetInputType = _Simple_GetInputType;
    op->base_op.GetOutputTypeCount = _Simple_GetOutputTypeCount;
    op->base_op.GetOutputType = _Simple_GetOutputType;
    op->base_op.CreateKernel = _Simple_CreateKernel;
    op->base_op.KernelCompute = _Simple_KernelCompute;
    op->base_op.KernelDestroy = _Simple_KernelDestroy;
}
