#include <stdio.h>

#include "weights.h"

#include "adci_tensor.h"
#include "adci_tensor_op.h"
#include "adci_graph.h"

void conv_layer(
    struct adci_tensor *input,
    struct adci_tensor *padding, 
    struct adci_tensor *filter,
    unsigned int *stride_data,
    unsigned int *dims_data,
    struct adci_tensor *bias,
    struct adci_tensor *output)
    {
    /* PADDING LAYER*/
    if(padding){
        struct adci_vector inputs = adci_vector_init(sizeof(struct adci_tensor *));
        adci_vector_add(&inputs, &input);
        adci_vector_add(&inputs, &padding);
        adci_tensor_pad(inputs, output);
        adci_vector_free(&inputs);
    }

    /* CONV LAYER */
    {
        struct adci_tensor *stride = adci_tensor_init_1d(2, ADCI_I32);
        adci_tensor_alloc_set(stride, stride_data);
        struct adci_tensor *dims = adci_tensor_init_1d(2, ADCI_I32);
        adci_tensor_alloc_set(dims, dims_data);
        struct adci_vector inputs = adci_vector_init(sizeof(struct adci_tensor *));
        adci_vector_add(&inputs, &output);
        adci_vector_add(&inputs, &filter);
        adci_vector_add(&inputs, &stride);
        adci_vector_add(&inputs, &dims);
        adci_tensor_conv2D(inputs, output);
        inputs.length = 0;
        adci_tensor_free(dims);
        adci_tensor_free(stride);
        adci_tensor_free(filter);

    /* BIAS LAYER */
        adci_vector_add(&inputs, &output);
        adci_vector_add(&inputs, &bias);
        adci_tensor_add(inputs, output);
        adci_vector_free(&inputs);
        adci_tensor_free(bias);
    }
}

int main(void){
    printf("ADCI IMPLEMENTATION FOR DIGIT RECOGNIZER \n");

    unsigned int input_shape[] = {1, 28, 28, 1};
    struct adci_tensor *input = adci_tensor_init(4, input_shape, ADCI_F32);
    adci_tensor_alloc(input);
    adci_tensor_fill(input, (float[]){0.f});

    /* COMMON TENSORS */
    unsigned int padding_values[][2] = { {0, 0}, {1, 1}, {1, 1}, {0, 0} };
    struct adci_tensor *padding = adci_tensor_init_2d(4, 2, ADCI_I32);
    adci_tensor_alloc_set(padding, padding_values);

    /* MAIN OUTPUT TENSOR */
    struct adci_tensor output;
    memset(&output, 0, sizeof(struct adci_tensor));
    
    /* CONV2D_0 */
    {
        unsigned int filter_shape[] = {32, 3, 3, 1};
        struct adci_tensor *filter = adci_tensor_init(4, filter_shape, ADCI_F32);
        adci_tensor_alloc_set(filter, conv2D_filter);
        struct adci_tensor *bias = adci_tensor_init_1d(filter_shape[0], ADCI_F32);
        adci_tensor_alloc_set(bias, conv2D_bias);
        conv_layer(input, padding, filter, (unsigned int[]){1, 1}, (unsigned int[]){1, 2}, bias, &output);
    }

    adci_tensor_print_shape(&output);

    /* RELU ACTIVATION LAYER */
    {
        struct adci_vector inputs = adci_vector_init(sizeof(struct adci_tensor *));
        adci_vector_add(&inputs, (struct adci_tensor*[]){&output});
        adci_tensor_relu(inputs, &output);
        adci_vector_free(&inputs);
    }

    adci_tensor_print_shape(&output);

    /* MULT LAYER */
    {
        struct adci_tensor *operand = adci_tensor_init_1d(32, ADCI_F32);
        adci_tensor_alloc_set(operand, mult_weights);
        struct adci_vector inputs = adci_vector_init(sizeof(struct adci_tensor *));
        adci_vector_add(&inputs, (struct adci_tensor*[]){&output});
        adci_vector_add(&inputs, &operand);
        adci_tensor_mul(inputs, &output);
        adci_vector_free(&inputs);
        adci_tensor_free(operand);
    }

    adci_tensor_print_shape(&output);

    /* ADD LAYER */
    {
        struct adci_tensor *operand = adci_tensor_init_1d(32, ADCI_F32);
        adci_tensor_alloc_set(operand, add_weights);
        struct adci_vector inputs = adci_vector_init(sizeof(struct adci_tensor *));
        adci_vector_add(&inputs, (struct adci_tensor*[]){&output});
        adci_vector_add(&inputs, &operand);
        adci_tensor_add(inputs, &output);
        adci_vector_free(&inputs);
        adci_tensor_free(operand);
    }

    adci_tensor_print_shape(&output);

    /* CONV2D_1 */
    {
        unsigned int filter_shape[] = {64, 3, 3, 32};
        struct adci_tensor *filter = adci_tensor_init(4, filter_shape, ADCI_F32);
        adci_tensor_alloc_set(filter, conv2D_filter_1);
        struct adci_tensor *bias = adci_tensor_init_1d(filter_shape[0], ADCI_F32);
        adci_tensor_alloc_set(bias, conv2D_bias_1);
        conv_layer(&output, padding, filter, (unsigned int[]){1, 1}, (unsigned int[]){1, 2}, bias, &output);
    }

    adci_tensor_print_shape(&output);

    /* MAX_POOL2D_0 */
    {
        struct adci_tensor *size = adci_tensor_init_1d(2, ADCI_I32);
        adci_tensor_alloc_set(size, (unsigned int[]){2, 2});
        struct adci_tensor *stride = adci_tensor_init_1d(2, ADCI_I32);
        adci_tensor_alloc_set(stride, (unsigned int[]){2, 2});
        struct adci_tensor *dims = adci_tensor_init_1d(2, ADCI_I32);
        adci_tensor_alloc_set(dims, (unsigned int[]){1, 2});
        struct adci_vector inputs = adci_vector_init(sizeof(struct adci_tensor *));
        adci_vector_add(&inputs, (struct adci_tensor*[]){&output});
        adci_vector_add(&inputs, &size);
        adci_vector_add(&inputs, &stride);
        adci_vector_add(&inputs, &dims);
        adci_tensor_max_pool2D(inputs, &output);
        adci_vector_free(&inputs);
        adci_tensor_free(dims);
        adci_tensor_free(stride);
        adci_tensor_free(size);
    }

    adci_tensor_print_shape(&output);

    /* CONV2D_2 */
    {
        unsigned int filter_shape[] = {128, 3, 3, 64};
        struct adci_tensor *filter = adci_tensor_init(4, filter_shape, ADCI_F32);
        adci_tensor_alloc_set(filter, conv2D_filter_2);
        struct adci_tensor *bias = adci_tensor_init_1d(filter_shape[0], ADCI_F32);
        adci_tensor_alloc_set(bias, conv2D_bias_2);
        conv_layer(&output, padding, filter, (unsigned int[]){1, 1}, (unsigned int[]){1, 2}, bias, &output);
    }

    adci_tensor_print_shape(&output);

    /* MAX_POOL2D_1 */
    {
        struct adci_tensor *size = adci_tensor_init_1d(2, ADCI_I32);
        adci_tensor_alloc_set(size, (unsigned int[]){2, 2});
        struct adci_tensor *stride = adci_tensor_init_1d(2, ADCI_I32);
        adci_tensor_alloc_set(stride, (unsigned int[]){2, 2});
        struct adci_tensor *dims = adci_tensor_init_1d(2, ADCI_I32);
        adci_tensor_alloc_set(dims, (unsigned int[]){1, 2});
        struct adci_vector inputs = adci_vector_init(sizeof(struct adci_tensor *));
        adci_vector_add(&inputs, (struct adci_tensor*[]){&output});
        adci_vector_add(&inputs, &size);
        adci_vector_add(&inputs, &stride);
        adci_vector_add(&inputs, &dims);
        adci_tensor_max_pool2D(inputs, &output);
        adci_vector_free(&inputs);
        adci_tensor_free(dims);
        adci_tensor_free(stride);
        adci_tensor_free(size);
    }

    adci_tensor_print_shape(&output);

    /* TRANSPOSE LAYER */
    {
        struct adci_tensor *dims = adci_tensor_init_1d(4, ADCI_I32);
        adci_tensor_alloc_set(dims, (unsigned int[]){0, 3, 1, 2});
        struct adci_vector inputs = adci_vector_init(sizeof(struct adci_tensor *));
        adci_vector_add(&inputs, (struct adci_tensor*[]){&output});
        adci_vector_add(&inputs, &dims);
        adci_tensor_transpose(inputs, &output);
        adci_vector_free(&inputs);
        adci_tensor_free(dims);
    }

    adci_tensor_print_shape(&output);

    /* RESHAPE LAYER */
    {
        struct adci_tensor *shape = adci_tensor_init_1d(2, ADCI_I32);
        adci_tensor_alloc_set(shape, (unsigned int[]){1, 6272});
        struct adci_vector inputs = adci_vector_init(sizeof(struct adci_tensor *));
        adci_vector_add(&inputs, (struct adci_tensor*[]){&output});
        adci_vector_add(&inputs, &shape);
        adci_tensor_reshape(inputs, &output);
        adci_vector_free(&inputs);
        adci_tensor_free(shape);
    }

    adci_tensor_print_shape(&output);

    /* FULLY_CONNECTED_0 LAYER*/
    {
        unsigned int shape[] = {384, 6272};
        struct adci_tensor *weights = adci_tensor_init(2, shape, ADCI_F32);
        adci_tensor_alloc_set(weights, fullyconnected_weights);
        struct adci_vector inputs = adci_vector_init(sizeof(struct adci_tensor *));
        adci_vector_add(&inputs, (struct adci_tensor*[]){&output});
        adci_vector_add(&inputs, &weights);
        adci_tensor_fully_connected(inputs, &output);
        adci_tensor_free(weights);
        inputs.length = 0;

        /* BIAS */
        struct adci_tensor *bias = adci_tensor_init_1d(shape[0], ADCI_F32);
        adci_tensor_alloc_set(weights, fullyconnected_bias);
        adci_vector_add(&inputs, (struct adci_tensor*[]){&output});
        adci_vector_add(&inputs, &bias);
        adci_tensor_add(inputs, &output);
        adci_vector_free(&inputs);
        adci_tensor_free(bias);
    }

    adci_tensor_print_shape(&output);

    /* RELU ACTIVATION LAYER */
    {
        struct adci_vector inputs = adci_vector_init(sizeof(struct adci_tensor *));
        adci_vector_add(&inputs, (struct adci_tensor*[]){&output});
        adci_tensor_relu(inputs, &output);
        adci_vector_free(&inputs);
    }

    adci_tensor_print_shape(&output);

    /* FULLY_CONNECTED_1 LAYER*/
    {
        unsigned int shape[] = {10, 384};
        struct adci_tensor *weights = adci_tensor_init(2, shape, ADCI_F32);
        adci_tensor_alloc_set(weights, fullyconnected_weights_1);
        struct adci_vector inputs = adci_vector_init(sizeof(struct adci_tensor *));
        adci_vector_add(&inputs, (struct adci_tensor*[]){&output});
        adci_vector_add(&inputs, &weights);
        adci_tensor_fully_connected(inputs, &output);
        adci_tensor_free(weights);
        inputs.length = 0;    

        /* BIAS */
        struct adci_tensor *bias = adci_tensor_init_1d(shape[0], ADCI_F32);
        adci_tensor_alloc_set(bias, fullyconnected_bias_1);
        adci_vector_add(&inputs, (struct adci_tensor*[]){&output});
        adci_vector_add(&inputs, &bias);
        adci_tensor_add(inputs, &output);
        adci_vector_free(&inputs);
        adci_tensor_free(bias);
    }

    adci_tensor_print_shape(&output);

    /* SOFTMAX LAYER */
    {
        struct adci_tensor *dims = adci_tensor_init_1d(1, ADCI_I32);
        adci_tensor_alloc_set(dims, (unsigned int[]){1});
        struct adci_vector inputs = adci_vector_init(sizeof(struct adci_tensor *));
        adci_vector_add(&inputs, (struct adci_tensor*[]){&output});
        adci_vector_add(&inputs, &dims);
        adci_tensor_softmax(inputs, &output);
        adci_vector_free(&inputs);
        adci_tensor_free(dims);
    }

    adci_tensor_print_shape(&output);

    /* OUTPUT READY */
    printf("COMPUTATION DONE FOR DIGIT RECOGNITION\n");

    /* CLEAN UP COMMON TENSORS */
    adci_tensor_free(input);
    adci_tensor_free(padding);
    ADCI_FREE(output.data);

    return 0;
}