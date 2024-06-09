#ifndef ADCI_TENSOR_OP_H
#define ADCI_TENSOR_OP_H

#include "adci_tensor.h"

enum adci_tensor_op{
    ADCI_TENSOR_COPY,
    ADCI_TENSOR_ADD,
    ADCI_TENSOR_SUB,
    ADCI_TENDOR_MUL,
    ADCI_TENSOR_BATCH_MATMUL,
    ADCI_TENSOR_PAD,
    ADCI_TENSOR_CONV2D,
    ADCI_TENSOR_PRELU,
    ADCI_TENSOR_CONCAT,
    ADCI_TENSOR_AVG_POOL2D,
    ADCI_TENSOR_TRANSPOSE,
    ADCI_TENSOR_RESHAPE,
    ADCI_TENSOR_REDUCE_MAX,
    ADCI_TENSOR_SOFTMAX,
    ADCI_TENSOR_TRANSPOSE_CONV
};

void ADCI_EXIT_POINT adci_tensor_add(struct adci_tensor **inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_reshape(struct adci_tensor *input, unsigned int *shape, unsigned int n_dims);

void ADCI_EXIT_POINT adci_tensor_compute_op(struct adci_tensor **inputs, struct adci_tensor *output, enum adci_tensor_op op);

#endif //ADCI_TENSOR_OP_H