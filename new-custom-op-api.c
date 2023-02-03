// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdbool.h>

#include "new-custom-op-api.h"

struct OrtSimpleCustomOp {
    struct OrtCustomOp base_op;
    const OrtApi* ort;
    OrtAllocator* allocator;
    const OrtSimpleCustomOpConfig config;
};

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
    return ((struct OrtSimpleCustomOp*)op)->config.name;
}

static size_t _Simple_GetInputTypeCount(const OrtCustomOp* op)
{
    return ((struct OrtSimpleCustomOp*)op)->config.inputs.count;
}

static ONNXTensorElementDataType _Simple_GetInputType(const OrtCustomOp* op, size_t index)
{
    const OrtSimpleCustomOpIOConfig config = ((struct OrtSimpleCustomOp*)op)->config.inputs;
    if (config.heterogeneous_types != NULL) {
        return config.heterogeneous_types[index];
    } else {
        return config.homogenous_type;
    }
}

static size_t _Simple_GetOutputTypeCount(const OrtCustomOp* op)
{
    return ((struct OrtSimpleCustomOp*)op)->config.outputs.count;
}

static ONNXTensorElementDataType _Simple_GetOutputType(const OrtCustomOp* op, size_t index)
{
    const OrtSimpleCustomOpIOConfig config = ((struct OrtSimpleCustomOp*)op)->config.outputs;
    if (config.heterogeneous_types != NULL) {
        return config.heterogeneous_types[index];
    } else {
        return config.homogenous_type;
    }
}

static void* _Simple_CreateKernel(const OrtCustomOp* op, const OrtApi* ort, const OrtKernelInfo* info)
{
    return (void*)op;
}

static void _Simple_KernelCompute(void* kernel, OrtKernelContext* context)
{
    const struct OrtSimpleCustomOp* op = (struct OrtSimpleCustomOp*)kernel;
    op->config.kernel_compute(op, op->ort, context);
}

static void _Simple_KernelDestroy(void* kernel)
{
}

ORT_EXPORT OrtStatus* OrtCreateSimpleCustomOp(
    _In_ const OrtApi* api,
    _In_ OrtAllocator* allocator,
    _In_ const OrtSimpleCustomOpConfig* config,
    _Out_ OrtSimpleCustomOp** op)
{
    OrtStatus *ort_status = NULL;

    if (allocator == NULL) {
        if ((ort_status = api->GetAllocatorWithDefaultOptions(&allocator)) != NULL) {
            return ort_status;
        }
    }

    if ((ort_status = api->AllocatorAlloc(
        allocator,
        sizeof(struct OrtSimpleCustomOp),
        (void**)op)) != NULL) {
        return ort_status;
    }

    OrtSimpleCustomOp* _op = *op;

    _op->ort = api;
    _op->allocator = allocator;

    memcpy((void*)&_op->config, (void*)config, sizeof(struct OrtSimpleCustomOpConfig));

    OrtInitializeCustomOpDefaults(&_op->base_op);
    _op->base_op.GetName = _Simple_GetName;
    _op->base_op.GetInputTypeCount = _Simple_GetInputTypeCount;
    _op->base_op.GetInputType = _Simple_GetInputType;
    _op->base_op.GetOutputTypeCount = _Simple_GetOutputTypeCount;
    _op->base_op.GetOutputType = _Simple_GetOutputType;
    _op->base_op.CreateKernel = _Simple_CreateKernel;
    _op->base_op.KernelCompute = _Simple_KernelCompute;
    _op->base_op.KernelDestroy = _Simple_KernelDestroy;

    return NULL;
}

OrtStatus* OrtSimpleCustomOpIORelease(
    _In_ const OrtSimpleCustomOp *op,
    _In_ OrtSimpleCustomOpIO* io)
{
    if (io != NULL && io->dims != NULL) {
        OrtStatus *ort_status = op->ort->AllocatorFree(op->allocator, (void*)io->dims);
        if (ort_status != NULL) {
            return ort_status;
        }
        io->dims = NULL;
    }
    return NULL;
}

static OrtStatus* OrtSimpleCustomOpReadOpIO(
    _In_ const OrtSimpleCustomOp* op,
    _In_ const OrtKernelContext* context,
    _In_ size_t index,
    _Out_ OrtSimpleCustomOpIO* io)
{
    OrtStatus* ort_status;
    const OrtApi *ort = op->ort;

    io->index = index;

    if ((ort_status = ort->GetTensorMutableData(
        (OrtValue*)io->value,
        &io->buffer)) != NULL) {
        return ort_status;
    }

    if ((ort_status = ort->GetTensorTypeAndShape(
        io->value,
        &io->type_and_shape_info)) != NULL) {
        return ort_status;
    }

    if ((ort_status = ort->GetTensorShapeElementCount(
        io->type_and_shape_info,
        &io->buffer_len)) != NULL) {
        return ort_status;
    }

    if ((ort_status = ort->GetDimensionsCount(
        (const OrtTensorTypeAndShapeInfo*)io->type_and_shape_info,
        &io->dims_len)) != NULL) {
        return ort_status;
    }

    if ((ort_status = ort->AllocatorAlloc(
        op->allocator,
        io->dims_len,
        (void**)&io->dims)) != NULL) {
        return ort_status;
    }

    if ((ort_status = ort->GetDimensions(
        (const OrtTensorTypeAndShapeInfo*)io->type_and_shape_info,
        io->dims,
        io->dims_len)) != NULL) {
        return ort_status;
    }

    return NULL;
}

OrtStatus* OrtSimpleCustomOpGetInput(
    _In_ const OrtSimpleCustomOp* op,
    _In_ const OrtKernelContext* context,
    _In_ size_t index,
    _Out_ OrtSimpleCustomOpIO* input)
{
    OrtStatus* ort_status;

    if ((ort_status = op->ort->KernelContext_GetInput(
        context,
        index,
        &input->value)) != NULL) {
        return ort_status;
    }

    return OrtSimpleCustomOpReadOpIO(op, context, index, input);
}

OrtStatus* OrtSimpleCustomOpGetOutput(
    _In_ const OrtSimpleCustomOp* op,
    _In_ const OrtKernelContext* context,
    _In_ size_t index,
    _In_ const int64_t* dims,
    _In_ size_t dims_len,
    _Out_ OrtSimpleCustomOpIO* input)
{
    OrtStatus* ort_status;

    if ((ort_status = op->ort->KernelContext_GetOutput(
        (OrtKernelContext*)context,
        index,
        dims,
        dims_len,
        (OrtValue**)&input->value)) != NULL) {
        return ort_status;
    }

    return OrtSimpleCustomOpReadOpIO(op, context, index, input);
}

OrtStatus* OrtSimpleCustomOpRegister(
    _In_ const OrtApi* ort,
    _In_ OrtAllocator* allocator,
    _In_ const char* custom_op_domain_name,
    _In_ const OrtSimpleCustomOpConfig* custom_op_configs,
    _In_ size_t custom_op_configs_len,
    _Out_ OrtCustomOpDomain** custom_op_domain,
    _Out_ OrtSimpleCustomOp** custom_ops)
{
    OrtStatus *ort_status = NULL;

    if ((ort_status = ort->CreateCustomOpDomain(
        custom_op_domain_name,
        custom_op_domain)) != NULL) {
        return ort_status;
    }

    for (size_t i = 0; i < custom_op_configs_len; i++) {
        OrtSimpleCustomOp *custom_op;
        if ((ort_status = OrtCreateSimpleCustomOp(
            ort,
            allocator,
            &custom_op_configs[i],
            &custom_op)) != NULL) {
            return ort_status;
        }

        if (custom_ops != NULL) {
            custom_ops[i] = custom_op;
        }

        if ((ort_status = ort->CustomOpDomain_Add(
            *custom_op_domain,
            (OrtCustomOp*)custom_op)) != NULL) {
            return ort_status;
        }
    }

    return NULL;
}
