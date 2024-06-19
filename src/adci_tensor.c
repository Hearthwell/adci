#include "adci_common.h"
#include "adci_tensor.h"
#include "adci_logging.h"

/* PRIVATE FUNCTIONS */

ADCI_TEST_VISIBILITY unsigned int adci_tensor_element_count_ext(unsigned int n_dims, unsigned int *shape){
    unsigned int size = 1;
    for(unsigned int i = 0; i < n_dims; i++)
        size *= shape[i];
    return size;
}

/* END PRIVATE FUNCTIONS */

unsigned int adci_tensor_element_count(struct adci_tensor *tensor){
    return adci_tensor_element_count_ext(tensor->n_dimension, tensor->shape);
}

struct adci_tensor * adci_tensor_init(unsigned int n_dims, const unsigned int *shape, enum adci_tensor_type type){
    struct adci_tensor *tensor = (struct adci_tensor *) ADCI_ALLOC(sizeof(struct adci_tensor));
    tensor->n_dimension = n_dims;
    tensor->data = NULL;
    for(unsigned int i = 0; i < tensor->n_dimension; i++)
        tensor->shape[i] = shape[i];
    tensor->dtype = type;
    return tensor;
}

struct adci_tensor * adci_tensor_init_1d(unsigned int count, enum adci_tensor_type type){
    struct adci_tensor *tensor = (struct adci_tensor *) ADCI_ALLOC(sizeof(struct adci_tensor));
    tensor->n_dimension = 1;
    tensor->data = NULL;
    tensor->shape[0] = count;
    tensor->dtype = type;
    return tensor;
}

struct adci_tensor * adci_tensor_init_2d(unsigned int dim1, unsigned int dim2, enum adci_tensor_type type){
    struct adci_tensor *tensor = (struct adci_tensor *) ADCI_ALLOC(sizeof(struct adci_tensor));
    tensor->n_dimension = 2;
    tensor->data = NULL;
    tensor->shape[0] = dim1;
    tensor->shape[1] = dim2;
    tensor->dtype = type;
    return tensor;
}

void adci_tensor_alloc(struct adci_tensor *tensor){
    tensor->data = ADCI_ALLOC(adci_tensor_element_count(tensor) * adci_tensor_dtype_size(tensor->dtype));
}

void adci_tensor_alloc_set(struct adci_tensor *tensor, const void *data){
    adci_tensor_alloc(tensor);
    adci_tensor_set(tensor, data);
}

void adci_tensor_free(struct adci_tensor *tensor){
    ADCI_FREE(tensor->data);
    tensor->data = NULL;
    ADCI_FREE(tensor);
}

unsigned int adci_tensor_set(struct adci_tensor *tensor, const void *data){
    if(tensor->data == NULL){
        ADCI_LOG(ADCI_ERROR, "TRYING TO SET UNALLOCATED TENSOR, ABORT");
        return 0;
    }
    const unsigned int size = adci_tensor_element_count(tensor) * adci_tensor_dtype_size(tensor->dtype);
    memcpy(tensor->data, data, size);
    return size;
}

struct adci_tensor * adci_tensor_get_view(struct adci_tensor *src, unsigned int n_index, unsigned int *index){
    ADCI_ASSERT(n_index <= src->n_dimension);
    ADCI_ASSERT(src->data != NULL);
    const unsigned int n_dims = src->n_dimension - n_index;
    struct adci_tensor *view = adci_tensor_init(n_dims, src->shape + n_dims, src->dtype);
    unsigned int offset = 0;
    for(unsigned int i = 0; i < n_index; i++){
        const unsigned int block_size = adci_tensor_element_count_ext(src->n_dimension - i - 1, src->shape + i + 1) * adci_tensor_dtype_size(view->dtype);
        offset += index[i] * block_size;
    }
    view->data = (uint8_t *)src->data + offset;
    return view;
}

bool adci_tensor_clean_view(struct adci_tensor *view){
    view->data = NULL;
    ADCI_FREE(view);
    return true;
}

unsigned int adci_tensor_dtype_size(enum adci_tensor_type dtype){
    switch (dtype){
    case ADCI_F32: return sizeof(float);
    case ADCI_I32: return sizeof(int32_t);
    case ADCI_I8:  return sizeof(int8_t); 
    }
    ADCI_ASSERT("SHOULD NEVER REACH" == 0);
    return 0;
}