// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

static void random_fill(float* buffer, size_t buffer_size)
{
    for (size_t i = 0; i < buffer_size; i++) {
        buffer[i] = (float)rand() / (float)RAND_MAX;
    }
}

static void custom_op1_kernel(
    const OrtSimpleCustomOp* op,
    const OrtApi* ort,
    const OrtKernelContext* context)
{
    OrtSimpleCustomOpIO input_x;
    OrtSimpleCustomOpGetInput(op, context, 0, &input_x);

    OrtSimpleCustomOpIO input_y;
    OrtSimpleCustomOpGetInput(op, context, 1, &input_y);

    OrtSimpleCustomOpIO output;
    OrtSimpleCustomOpGetOutput(op, context, 0, input_x.dims, input_x.dims_len, &output);

    float* input_x_buffer = (float*)input_x.buffer;
    float* input_y_buffer = (float*)input_y.buffer;
    float* output_buffer = (float*)output.buffer;

    for (size_t i = 0; i < input_x.buffer_len; i++) {
        output_buffer[i] = input_x_buffer[i] + input_y_buffer[i];
    }

    OrtSimpleCustomOpIORelease(op, &input_x);
    OrtSimpleCustomOpIORelease(op, &input_y);
    OrtSimpleCustomOpIORelease(op, &output);
}

static void custom_op2_kernel(
    const OrtSimpleCustomOp* op,
    const OrtApi* ort,
    const OrtKernelContext* context)
{
    OrtSimpleCustomOpIO input;
    OrtSimpleCustomOpGetInput(op, context, 0, &input);

    OrtSimpleCustomOpIO output;
    OrtSimpleCustomOpGetOutput(op, context, 0, input.dims, input.dims_len, &output);

    float* input_buffer = (float*)input.buffer;
    int32_t* output_buffer = (int32_t*)output.buffer;

    for (size_t i = 0; i < input.buffer_len; i++) {
        output_buffer[i] = round(input_buffer[i]);
    }

    OrtSimpleCustomOpIORelease(op, &input);
    OrtSimpleCustomOpIORelease(op, &output);
}

static OrtSimpleCustomOpConfig custom_ops[] = {
    {
        .name = "CustomOpOne",
        .inputs = {
            .count = 2,
            .homogenous_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        },
        .outputs = {
            .count = 1,
            .homogenous_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        },
        .kernel_compute = custom_op1_kernel
    },
    {
        .name = "CustomOpTwo",
        .inputs = {
            .count = 1,
            .homogenous_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        },
        .outputs = {
            .count = 1,
            .homogenous_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        },
        .kernel_compute = custom_op2_kernel
    }
};

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

    OrtSessionOptions* ort_session_options;
    if ((ort_status = ort->CreateSessionOptions(&ort_session_options)) != NULL) {
        return ort_error(ort, ort_status);
    }

    OrtCustomOpDomain* custom_op_domain;
    if ((ort_status = OrtSimpleCustomOpRegister(
        ort,
        NULL /* allocator */,
        "test.customop",
        custom_ops,
        sizeof(custom_ops) / sizeof(custom_ops[0]),
        &custom_op_domain,
        NULL /* custom_ops */)) != NULL) {
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
