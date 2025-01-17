#ifndef ADCI_TENSOR_OP_H
#define ADCI_TENSOR_OP_H

#include "adci_tensor.h"

enum adci_tensor_op{
    ADCI_TENSOR_INPUT,
    ADCI_TENSOR_COPY,
    ADCI_TENSOR_ADD,
    ADCI_TENSOR_SUB,
    ADCI_TENSOR_RESHAPE,
    ADCI_TENSOR_PAD,
    ADCI_TENSOR_PRELU,
    ADCI_TENSOR_CAST,
    ADCI_TENSOR_SOFTMAX,
    ADCI_TENSOR_REDUCE_MAX,
    ADCI_TENSOR_CONCAT,
    ADCI_TENSOR_MUL,
    ADCI_TENSOR_MAX_POOL2D,
    ADCI_TENSOR_RELU,
    ADCI_TENSOR_CONV2D,
    ADCI_TENSOR_TRANSPOSE,
    ADCI_TENSOR_FULLY_CONNECTED,
    ADCI_TENSOR_BATCH_MATMUL,
    ADCI_TENSOR_AVG_POOL2D,
    ADCI_TENSOR_ARGMAX,

    ADCI_TENSOR_TRANSPOSE_CONV,
};

/* SHOULD BUILD COMPUTE GRAPH INSTEAD OF USING INDIVIDUAL TENSOR OPS FOR NN */
void ADCI_EXIT_POINT adci_tensor_add(struct adci_vector inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_sub(struct adci_vector inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_reshape(struct adci_vector inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_pad(struct adci_vector inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_prelu(struct adci_vector inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_relu(struct adci_vector inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_cast(struct adci_vector inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_softmax(struct adci_vector inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_reduce_max(struct adci_vector inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_concat(struct adci_vector inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_mul(struct adci_vector inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_max_pool2D(struct adci_vector inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_avg_pool2D(struct adci_vector inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_conv2D(struct adci_vector inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_transpose(struct adci_vector inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_fully_connected(struct adci_vector inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_copy(struct adci_tensor *input, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_batch_matmult(struct adci_vector inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_argmax(struct adci_vector inputs, struct adci_tensor *output);

/* EXTENDED ARGS VERSION OF PREVIOUS OPS */
void ADCI_EXIT_POINT adci_tensor_relu_args(
    struct adci_tensor *tensor, 
    struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_prelu_args(
    struct adci_tensor *element,
    struct adci_tensor *parameters, 
    struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_reduce_max_args(
    struct adci_tensor *tensor,
    struct adci_tensor *axis, 
    struct adci_tensor *keepdims, 
    struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_max_pool2D_args(
    struct adci_tensor *tensor, 
    struct adci_tensor *size, 
    struct adci_tensor *stride,
    struct adci_tensor *dims,
    struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_avg_pool2D_args(
    struct adci_tensor *tensor, 
    struct adci_tensor *size, 
    struct adci_tensor *stride,
    struct adci_tensor *dims,
    struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_conv2D_args(
    struct adci_tensor *tensor,
    struct adci_tensor *filter,
    struct adci_tensor *stride,
    struct adci_tensor *dims,
    struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_transpose_args(
    struct adci_tensor *tensor, 
    struct adci_tensor *dims, 
    struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_batch_matmult_args(
    struct adci_tensor *first, 
    struct adci_tensor *second, 
    struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_argmax_args(
    struct adci_tensor *tensor,
    struct adci_tensor *dim,
    struct adci_tensor *keep_dim,
    struct adci_tensor *output);

void ADCI_EXIT_POINT adci_tensor_compute_op(struct adci_vector inputs, struct adci_tensor *output, enum adci_tensor_op op);
void ADCI_EXIT_POINT adci_tensor_compute_op_shape(struct adci_vector inputs, struct adci_tensor *output, enum adci_tensor_op op);

const char * adci_tensor_op_str(enum adci_tensor_op op);
#endif //ADCI_TENSOR_OP_H