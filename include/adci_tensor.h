#ifndef ADCI_TENSOR_H
#define ADCI_TENSOR_H

#define ADCI_TENSOR_MAX_DIM 4 

#include "adci_common.h"

/* DONT CHANGE ORDER, INPORTANT FOR TENSOR_OP */
enum adci_tensor_type{
    ADCI_F32,
    ADCI_I32,
    ADCI_I8,
    /*
    ADCI_F16,
    ADCI_I4
    */
   ADCI_NUM_SUPPORTED_TYPES
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

struct adci_tensor * adci_tensor_init(unsigned int n_dims, const unsigned int *shape, enum adci_tensor_type type);
struct adci_tensor * adci_tensor_init_vargs(unsigned int n_dims, enum adci_tensor_type type, ...);
void adci_tensor_alloc(struct adci_tensor *tensor);
void adci_tensor_alloc_set(struct adci_tensor *tensor, const void *data);
void adci_tensor_free(struct adci_tensor *tensor);

/* RETURNS THE SIZE WRITTEN TO THE TENSOR */
unsigned int adci_tensor_set(struct adci_tensor *tensor, const void *data);
void adci_tensor_set_f32(struct adci_tensor *tensor, float element, ...);
void adci_tensor_set_i32(struct adci_tensor *tensor, int32_t element, ...);
void adci_tensor_set_element(struct adci_tensor *tensor, const void *element, ...);
void adci_tensor_fill(struct adci_tensor *tensor, const void *data);

void * adci_tensor_get_element(struct adci_tensor *tensor, ...);
float   adci_tensor_get_f32(const struct adci_tensor *tensor, ...);
int32_t adci_tensor_get_i32(const struct adci_tensor *tensor, ...);

/* RETURNS A VIEW ON THE SRC TENSOR, LETS DO STUFF LIKE, src[0][1] by giving n_index = 2 and index = {0, 1} */
struct adci_tensor * adci_tensor_get_view(struct adci_tensor *src, unsigned int n_index, unsigned int *index);
bool adci_tensor_clean_view(struct adci_tensor *view);

unsigned int adci_tensor_element_count(struct adci_tensor *tensor);
unsigned int adci_tensor_dtype_size(enum adci_tensor_type dtype);

const char * adci_tensor_dtype_str(enum adci_tensor_type dtype);

void adci_tensor_print(const struct adci_tensor *tensor);
void adci_tensor_print_shape(const struct adci_tensor *tensor);

#endif //ADCI_TENSOR_H