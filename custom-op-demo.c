#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#include <onnxruntime/core/session/onnxruntime_c_api.h>

#include "new-custom-op-api.h"

static int ort_error(const OrtApi* ort, OrtStatus* status)
{
    if (status != NULL) {
        fprintf(stderr, "ORT error: %s\n", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
    }

    return 1;
}

_Noreturn static void ort_abort(const OrtApi* ort, OrtStatus* status)
{
    ort_error(ort, status);
    abort();
}

static void random_fill(float* buffer, size_t buffer_size)
{
    for (size_t i = 0; i < buffer_size; i++) {
        buffer[i] = (float)rand() / (float)RAND_MAX;
    }
}

static void kernel_prepare_buffers(
    const OrtApi* ort,
    const OrtKernelContext* context,
    size_t *buffer_size,
    void **input_x_buffer,
    void **input_y_buffer,
    void **output_buffer)
{
    OrtStatus* ort_status = NULL;

    const OrtValue* input_x;
    if ((ort_status = ort->KernelContext_GetInput(context, 0, &input_x)) != NULL) {
        ort_abort(ort, ort_status);
    }

    if ((ort_status = ort->GetTensorMutableData((OrtValue*)input_x, input_x_buffer)) != NULL) {
        ort_abort(ort, ort_status);
    }

    if (input_y_buffer != NULL) {
        const OrtValue* input_y;
        if ((ort_status = ort->KernelContext_GetInput(context, 1, &input_y)) != NULL) {
            ort_abort(ort, ort_status);
        }

        if ((ort_status = ort->GetTensorMutableData((OrtValue*)input_y, input_y_buffer)) != NULL) {
            ort_abort(ort, ort_status);
        }
    }

    OrtTensorTypeAndShapeInfo* type_and_shape_info;
    if ((ort_status = ort->GetTensorTypeAndShape(input_x, &type_and_shape_info)) != NULL) {
        ort_abort(ort, ort_status);
    }

    size_t actual_dims_count;
    if ((ort_status = ort->GetDimensionsCount(
        (const OrtTensorTypeAndShapeInfo*)type_and_shape_info,
        &actual_dims_count)) != NULL) {
        ort_abort(ort, ort_status);
    }

    int64_t dims[2];
    const size_t dims_count = sizeof(dims) / sizeof(dims[0]);
    if ((ort_status = ort->GetDimensions(
        (const OrtTensorTypeAndShapeInfo*)type_and_shape_info,
        dims,
        dims_count)) != NULL) {
        ort_abort(ort, ort_status);
    }

    if (actual_dims_count != dims_count) {
        fprintf(stderr, "kernel1_compute: expected %ld dimensions, got %ld\n",
            dims_count, actual_dims_count);
        ort_abort(ort, NULL);
    }

    if ((ort_status = ort->GetTensorShapeElementCount(type_and_shape_info, buffer_size)) != NULL) {
        ort_abort(ort, ort_status);
    }

    OrtValue* output;
    if ((ort_status = ort->KernelContext_GetOutput(
        (OrtKernelContext*)context,
        0,
        dims,
        dims_count,
        &output)) != NULL) {
        ort_abort(ort, ort_status);
    }

    if ((ort_status = ort->GetTensorMutableData((OrtValue*)output, output_buffer)) != NULL) {
        ort_abort(ort, ort_status);
    }
}

static void kernel1_compute(const OrtApi* ort, const OrtKernelContext* context)
{
    size_t buffer_size;
    const float *input_x_buffer;
    const float *input_y_buffer;
    float *output_buffer;

    kernel_prepare_buffers(
        ort,
        context,
        &buffer_size,
        (void**)&input_x_buffer,
        (void**)&input_y_buffer,
        (void**)&output_buffer);

    for (size_t i = 0; i < buffer_size; i++) {
      output_buffer[i] = input_x_buffer[i] + input_y_buffer[i];
    }
}

static void kernel2_compute(const OrtApi* ort, const OrtKernelContext* context)
{
    size_t buffer_size;
    const float *input_buffer;
    int32_t *output_buffer;

    kernel_prepare_buffers(
        ort,
        context,
        &buffer_size,
        (void**)&input_buffer,
        NULL,
        (void**)&output_buffer);

    for (size_t i = 0; i < buffer_size; i++) {
      output_buffer[i] = round(input_buffer[i]);
    }
}

int main(int argc, const char* const* argv)
{
    srand((unsigned)time(NULL));

    OrtStatus *ort_status = NULL;
    const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    OrtEnv* ort_env;
    if ((ort_status = ort->CreateEnv(
        ORT_LOGGING_LEVEL_WARNING,
        "custom-op-demo",
        &ort_env)) != NULL) {
        return ort_error(ort, ort_status);
    }

    OrtCustomOpDomain* custom_op_domain;
    if ((ort_status = ort->CreateCustomOpDomain("test.customop", &custom_op_domain)) != NULL) {
        return ort_error(ort, ort_status);
    }

    OrtSimpleCustomOp custom_op1;
    ONNXTensorElementDataType custom_op1_input_types[] = {
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    };
    ONNXTensorElementDataType custom_op1_output_types[] = {
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    };
    OrtInitializeSimpleCustomOp(
        ort,
        "CustomOpOne",
        2, custom_op1_input_types,
        1, custom_op1_output_types,
        kernel1_compute,
        &custom_op1);

    if ((ort_status = ort->CustomOpDomain_Add(custom_op_domain, &custom_op1.base_op)) != NULL) {
        return ort_error(ort, ort_status);
    }

    OrtSimpleCustomOp custom_op2;
    ONNXTensorElementDataType custom_op2_input_types[] = {
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    };
    ONNXTensorElementDataType custom_op2_output_types[] = {
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
    };
    OrtInitializeSimpleCustomOp(
        ort,
        "CustomOpTwo",
        1, custom_op2_input_types,
        1, custom_op2_output_types,
        kernel2_compute,
        &custom_op2);

    if ((ort_status = ort->CustomOpDomain_Add(custom_op_domain, &custom_op2.base_op)) != NULL) {
        return ort_error(ort, ort_status);
    }

    OrtSessionOptions* ort_session_options;
    if ((ort_status = ort->CreateSessionOptions(&ort_session_options)) != NULL) {
        return ort_error(ort, ort_status);
    }

    if ((ort_status = ort->AddCustomOpDomain(ort_session_options, custom_op_domain)) != NULL) {
        return ort_error(ort, ort_status);
    }

    OrtSession* ort_session;
    if ((ort_status = ort->CreateSession(
        ort_env,
        "./custom_op_test.onnx",
        ort_session_options,
        &ort_session)) != NULL) {
        return ort_error(ort, ort_status);
    }

    OrtMemoryInfo* memory_info;
    if ((ort_status = ort->CreateCpuMemoryInfo(
        OrtArenaAllocator,
        OrtMemTypeDefault,
        &memory_info)) != NULL) {
        return ort_error(ort, ort_status);
    }

    const size_t buffer_size = 15;
    float input_x_buffer[buffer_size];
    float input_y_buffer[buffer_size];
    int32_t output_buffer[buffer_size];
    int32_t expected_output_buffer[buffer_size];
    const int64_t shape[] = { 3, 5 };
    const char* input_names[] = { "input_1", "input_2" };
    const char* output_names[] = { "output" };
    OrtValue *inputs[2];
    OrtValue *outputs[1];

    random_fill(input_x_buffer, buffer_size);
    random_fill(input_y_buffer, buffer_size);

    for (size_t i = 0; i < buffer_size; i++) {
        expected_output_buffer[i] = round(input_x_buffer[i] + input_y_buffer[i]);
    }

    if ((ort_status = ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        input_x_buffer,
        sizeof(input_x_buffer),
        shape,
        sizeof(shape) / sizeof(shape[0]),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &inputs[0])) != NULL) {
        return ort_error(ort, ort_status);
    }

    if ((ort_status = ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        input_y_buffer,
        sizeof(input_y_buffer),
        shape,
        sizeof(shape) / sizeof(shape[0]),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &inputs[1])) != NULL) {
        return ort_error(ort, ort_status);
    }

    if ((ort_status = ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        output_buffer,
        sizeof(output_buffer),
        shape,
        sizeof(shape) / sizeof(shape[0]),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
        &outputs[0])) != NULL) {
        return ort_error(ort, ort_status);
    }

    if ((ort_status = ort->Run(
        ort_session,
        NULL,
        input_names,
        (const OrtValue* const*)inputs,
        2,
        output_names,
        1,
        outputs)) != NULL) {
        return ort_error(ort, ort_status);
    }

    ort->ReleaseValue(inputs[0]);
    ort->ReleaseValue(inputs[1]);
    ort->ReleaseValue(outputs[0]);
    ort->ReleaseMemoryInfo(memory_info);
    ort->ReleaseSessionOptions(ort_session_options);
    ort->ReleaseSession(ort_session);
    ort->ReleaseEnv(ort_env);

    bool is_correct_result = true;
    for (size_t i = 0; i < buffer_size; i++) {
        if (output_buffer[i] != expected_output_buffer[i]) {
            is_correct_result = false;
            break;
        }
    }

    if (is_correct_result) {
        printf("success!\n");
        return 0;
    } else {
        fprintf(stderr, "output tensor does not match expectations\n");
        return 1;
    }
}
