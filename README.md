# ORT _Simple_ Custom Op API Experiment

The goal of this experiment is to provide a simple and easy to use C API for registering custom ONNX operators in ONNX Runtime with convenience APIs for scaffolding the kernel compute implementation.

### Example Custom Op

```c
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
```

### Example Custom Op Kernels

```c
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
```
