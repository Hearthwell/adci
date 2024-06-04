#ifndef ADCI_TENSOR_H
#define ADCI_TENSOR_H

#define ADCI_TENSOR_MAX_DIM 4 

#include "adci_common.h"

/* TODO, FOR NOW, ONLY SUPPORTED TYPE IS F32, ADD SUPPORT FOR OTHER TYPES (AT LEAST INT8) */
enum adci_tensor_type{
    ADCI_F32,
    /*
    ADCI_F16,
    ADCI_I8,
    ADCI_I4
    */
};

/* DIMESION ALWAYS STARTS AT INDEX 0, SO IN CASE OF TWO DIMS, WE GET [N, M, 0, 0] */
struct adci_tensor{
    unsigned int n_dimension;
    unsigned int shape[ADCI_TENSOR_MAX_DIM];
    enum adci_tensor_type dtype;
    /* TODO, ADD SUPPORT FOR NAME */
    struct adci_string *name;
    void *data;
};

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

struct adci_tensor * adci_tensor_init(unsigned int n_dims, const unsigned int *shape, enum adci_tensor_type type);
bool adci_tensor_alloc(struct adci_tensor *tensor);
bool adci_tensor_free(struct adci_tensor *tensor);

/* RETURNS THE SIZE WRITTEN TO THE TENSOR */
unsigned int adci_tensor_set(struct adci_tensor *tensor, const void *data);

/* RETURNS A VIEW ON THE SRC TENSOR, LETS DO STUFF LIKE, src[0][1] by giving n_index = 2 and index = {0, 1} */
struct adci_tensor * adci_tensor_get_view(struct adci_tensor *src, unsigned int n_index, unsigned int *index);
bool adci_tensor_clean_view(struct adci_tensor *view);

unsigned int adci_tensor_dtype_size(enum adci_tensor_type dtype);
struct adci_tensor * adci_tensor_compute_op(struct adci_tensor **inputs, struct adci_tensor *output, enum adci_tensor_op op);

#endif //ADCI_TENSOR_H