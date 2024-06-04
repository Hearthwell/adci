#include "adci_common.h"
#include "adci_tensor.h"

/* TODO, ADD SOME KIND OF MACRO TO DISABLE CHECKS TO SPEED UP INFERENCE */
static void adci_check_tensor_dim(struct adci_tensor **inputs){
    assert(inputs[0]->n_dimension == inputs[1]->n_dimension);
    for(unsigned int i = 0; i < inputs[0]->n_dimension; i++){
        assert(inputs[0]->shape[i] == inputs[1]->shape[i]);
    }
}

ADCI_TEST_VISIBILITY unsigned int adci_tensor_element_count_ext(unsigned int n_dims, unsigned int *shape){
    unsigned int size = 1;
    for(unsigned int i = 0; i < n_dims; i++)
        size *= shape[i];
    return size;
}

static unsigned int adci_tensor_element_count(struct adci_tensor *tensor){
    return adci_tensor_element_count_ext(tensor->n_dimension, tensor->shape);
}

ADCI_TEST_VISIBILITY struct adci_tensor * adci_compute_add(struct adci_tensor **inputs, struct adci_tensor *output){
    adci_check_tensor_dim(inputs);
    unsigned int tensor_volume = 1;
    for(unsigned int i = 0; i < inputs[0]->n_dimension; i++) tensor_volume *= inputs[0]->shape[i];
    #define ADD_FOR_TYPE(_type) ((_type*)output->data)[i] = ((_type*)inputs[0]->data)[i] + ((_type*)inputs[1]->data)[i]
    for(unsigned int i = 0; i < tensor_volume; i++){
        switch (inputs[0]->dtype){
            case ADCI_F32: 
                ADD_FOR_TYPE(float);
                break;
            /* TODO, HANDLE MORE TYPES */
        }
    }
    return output;
}

ADCI_TEST_VISIBILITY struct adci_tensor * adci_compute_reshape(struct adci_tensor *input, unsigned int *shape, unsigned int n_dims){
    const unsigned int required_count = adci_tensor_element_count(input);
    const unsigned int reshape_count = adci_tensor_element_count_ext(n_dims, shape);
    assert(required_count == reshape_count);
    for(unsigned int i = 0; i < n_dims; i++)
        input->shape[i] = shape[i];
    input->n_dimension = n_dims;
    return input;
}

/* END PRIVATE FUNCTIONS */

struct adci_tensor * adci_tensor_init(unsigned int n_dims, const unsigned int *shape, enum adci_tensor_type type){
    struct adci_tensor *tensor = (struct adci_tensor *) ADCI_ALLOC(sizeof(struct adci_tensor));
    tensor->n_dimension = n_dims;
    tensor->data = NULL;
    for(unsigned int i = 0; i < tensor->n_dimension; i++)
        tensor->shape[i] = shape[i];
    tensor->dtype = type;
    return tensor;
}

bool adci_tensor_alloc(struct adci_tensor *tensor){
    tensor->data = ADCI_ALLOC(adci_tensor_element_count(tensor) * adci_tensor_dtype_size(tensor->dtype));
    return tensor->data != NULL;
}

bool adci_tensor_free(struct adci_tensor *tensor){
    ADCI_FREE(tensor->data);
    tensor->data = NULL;
    ADCI_FREE(tensor);
    /* REMOVE AND MAKE FUNCTION VOID */
    return true;
}

unsigned int adci_tensor_set(struct adci_tensor *tensor, const void *data){
    const unsigned int size = adci_tensor_element_count(tensor) * adci_tensor_dtype_size(tensor->dtype);
    memcpy(tensor->data, data, size);
    return size;
}

struct adci_tensor * adci_tensor_get_view(struct adci_tensor *src, unsigned int n_index, unsigned int *index){
    assert(n_index <= src->n_dimension);
    assert(src->data != NULL);
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
    }
    assert("SHOULD NEVER REACH" == 0);
    return 0;
}

struct adci_tensor * adci_tensor_compute_op(struct adci_tensor **inputs, struct adci_tensor *output, enum adci_tensor_op op){
    switch (op){
    case ADCI_TENSOR_ADD: return adci_compute_add(inputs, output);
    case ADCI_TENSOR_RESHAPE: return adci_compute_reshape(inputs[0], inputs[1]->shape, adci_tensor_element_count(inputs[1]));
    default:
        assert("TODO, OPERATION NOT IMPLEMENTED YET" == 0);
    }
    assert("SHOULD NEVER REACH" == 0);
    return NULL;
}