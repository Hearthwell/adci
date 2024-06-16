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
    ADCI_TENSOR_TRANSPOSE_CONV,
    ADCI_TENSOR_INPUT
};

/* SHOULD NOT BE USED DIRECTLY NORMALY */
void ADCI_EXIT_POINT adci_tensor_add(struct adci_vector inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_reshape(struct adci_vector inputs, struct adci_tensor *output);
void ADCI_EXIT_POINT adci_tensor_copy(struct adci_tensor *input, struct adci_tensor *output);

void ADCI_EXIT_POINT adci_tensor_compute_op(struct adci_vector inputs, struct adci_tensor *output, enum adci_tensor_op op);
const char * adci_tensor_op_str(enum adci_tensor_op op);
#endif //ADCI_TENSOR_OP_H