#include <stdarg.h>
#include "adci_common.h"
#include "adci_tensor.h"
#include "adci_logging.h"

#include "adci_tensor_common.h"

/* PRIVATE FUNCTIONS */

static void adci_tensor_print_helper(const struct adci_tensor *input, unsigned int dim, unsigned int offset){
    /* TAB PREFIX */
    for(unsigned int i = 0; i < dim; i++) printf("\t");
    printf("[\n");
    if(dim == input->n_dimension - 1) 
        for(unsigned int i = 0; i < dim + 1; i++) printf("\t");
    const unsigned int volume = adci_tensor_element_count_ext(input->n_dimension - dim - 1, input->shape + dim + 1);
    for(unsigned int i = 0; i < input->shape[dim]; i++){
        if(dim == input->n_dimension - 1){
            switch (input->dtype){
            case ADCI_F32: printf("%4.2f, ", ((float *)input->data)[offset + i]);
            break;
            case ADCI_I32: printf("%d, ", ((int32_t *)input->data)[offset + i]);
            break;
            case ADCI_I8: printf("%d, ", ((int8_t *)input->data)[offset + i]);
            break;
            default: return;
            }
        }
        else adci_tensor_print_helper(input, dim + 1, offset + i * volume); 
    }

    if(dim == input->n_dimension - 1) printf("\n");
    /* TAB PREFIX */
    for(unsigned int i = 0; i < dim; i++) printf("\t");
    printf("],\n");
}

static void adci_tensor_set_element_helper(struct adci_tensor *tensor, const void *element, va_list ptr){
    unsigned int offset = 0; 
    for(unsigned int i = 0; i < tensor->n_dimension; i++){
        const unsigned int index = va_arg(ptr, unsigned int);
        const unsigned int volume = adci_tensor_element_count_ext(tensor->n_dimension - i - 1, tensor->shape + i + 1);   
        offset += index * volume;
    }
    const unsigned int element_size = adci_tensor_dtype_size(tensor->dtype);
    memcpy(tensor->data + offset * element_size, element, element_size);
    va_end(ptr);
}

static void * adci_tensor_get_element_helper(struct adci_tensor *tensor, va_list ptr){
    const unsigned int full_volume = adci_tensor_element_count_ext(tensor->n_dimension, tensor->shape);
    unsigned int volume = full_volume;
    unsigned int offset = 0;
    for(unsigned int i = 0; i < tensor->n_dimension; i++){
        volume /= tensor->shape[i];
        const unsigned int curr_offset = va_arg(ptr, unsigned int);
        offset += curr_offset * volume;
    }
    ADCI_ASSERT(offset < full_volume);
    va_end(ptr);
    return tensor->data + offset * adci_tensor_dtype_size(tensor->dtype);
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

void adci_tensor_set_f32(struct adci_tensor *tensor, float element, ...){
    ADCI_ASSERT(tensor->dtype == ADCI_F32);
    va_list ptr;
    va_start(ptr, element);
    adci_tensor_set_element_helper(tensor, &element, ptr);
}

void adci_tensor_set_i32(struct adci_tensor *tensor, int32_t element, ...){
    ADCI_ASSERT(tensor->dtype == ADCI_I32);
    va_list ptr;
    va_start(ptr, element);
    adci_tensor_set_element_helper(tensor, &element, ptr);
}

void adci_tensor_set_element(struct adci_tensor *tensor, const void *element, ...){
    va_list ptr;
    va_start(ptr, element);
    adci_tensor_set_element_helper(tensor, element, ptr);
}

void * adci_tensor_get_element(struct adci_tensor *tensor, ...){
    va_list ptr;
    va_start(ptr, tensor);
    return adci_tensor_get_element_helper(tensor, ptr);
}

float adci_tensor_get_f32(struct adci_tensor *tensor, ...){
    va_list ptr;
    va_start(ptr, tensor);
    return *(float *)adci_tensor_get_element_helper(tensor, ptr);
}

int32_t adci_tensor_get_i32(struct adci_tensor *tensor, ...){
    va_list ptr;
    va_start(ptr, tensor);
    return *(int32_t *)adci_tensor_get_element_helper(tensor, ptr);
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

void adci_tensor_print(const struct adci_tensor *tensor){
    adci_tensor_print_helper(tensor, 0, 0);
}