#include "adci_tensor_op.h"
#include "adci_logging.h"

/* PRIVATE FUNCTIONS */

static void ADCI_EXIT_POINT adci_check_tensor_types(struct adci_tensor **tensors){
    assert(tensors[0]->dtype == tensors[1]->dtype);
}

static void ADCI_EXIT_POINT adci_check_tensor_vec_types(struct adci_vector tensors){
    if(tensors.length == 0) return;
    enum adci_tensor_type type = (*(struct adci_tensor **)adci_vector_get(&tensors, 0))->dtype; 
    for(unsigned int i = 1; i < tensors.length; i++){
        enum adci_tensor_type current = (*(struct adci_tensor **)adci_vector_get(&tensors, i))->dtype; 
        assert(current == type);
    }
}

/* TODO, ADD SOME KIND OF MACRO TO DISABLE CHECKS TO SPEED UP INFERENCE */
static void ADCI_EXIT_POINT adci_check_tensor_dim(struct adci_tensor **inputs){
    assert(inputs[0]->n_dimension == inputs[1]->n_dimension);
    /* TODO, ADD LOGGING FOR TENSOR ERRORS */
    for(unsigned int i = 0; i < inputs[0]->n_dimension; i++){
        assert(inputs[0]->shape[i] == inputs[1]->shape[i]);
    }
}

static void adci_reset_value(void *data, enum adci_tensor_type type){
    #define ADCI_RESET(_type) *((_type*)data) = (_type)0
    switch (type){
    case ADCI_F32: ADCI_RESET(float);
    break;
    default:
        ADCI_LOG(ADCI_ERROR, "RESET FOR TYPE %d, NOT IMPLEMENTED", type);
    }
}

/* END PRIVATE FUNCTIONS */

void ADCI_EXIT_POINT adci_tensor_add(struct adci_vector inputs, struct adci_tensor *output){
    if(inputs.length <= 1){
        ADCI_LOG(ADCI_WARNING, "ADDITION OP WITH ONLY 1 TENSOR INPUT, COPYING TO OUTPUT");
        adci_tensor_copy(*(struct adci_tensor **)adci_vector_get(&inputs, 0), output);
        return;
    } 
    adci_check_tensor_dim(adci_vector_get(&inputs, 0));
    struct adci_tensor *first = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    adci_check_tensor_dim((struct adci_tensor *[]){first, output});
    unsigned int tensor_volume = 1;
    for(unsigned int i = 0; i < first->n_dimension; i++) tensor_volume *= first->shape[i];
    const unsigned int element_size = adci_tensor_dtype_size(output->dtype);
    #define ADD_FOR_TYPE(_first, _output, _type) ((_type*)_output->data)[i] += ((_type*)_first->data)[i]
    for(unsigned int i = 0; i < tensor_volume; i++){
        assert(output->data != NULL);
        adci_reset_value((uint8_t *)output->data + i * element_size, output->dtype);
        for(unsigned int j = 0; j < inputs.length; j++){
            struct adci_tensor *current = *(struct adci_tensor **)adci_vector_get(&inputs, j);
            assert(current->data != NULL);
            switch (current->dtype){
            case ADCI_F32: ADD_FOR_TYPE(current, output, float);
            break;
            case ADCI_I8: ADD_FOR_TYPE(current, output, uint8_t);
            break;
            default: ADCI_LOG(ADCI_ERROR, "ADD FOR TYPE %d, NOT IMPLEMENTED", current->dtype);
            break;
            }
        }
    }
}

/* FIRST ELEMENT IS THE TENSOR TO RESHAPE AND SECOND IS THE SHAPE TENSOR (DIM IS ONE)*/
void ADCI_EXIT_POINT adci_tensor_reshape(struct adci_vector inputs, struct adci_tensor *output){
    assert(inputs.length == 2);
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    struct adci_tensor *shape = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    assert(shape->n_dimension == 1);
    assert(shape->data != NULL);
    const unsigned int required_count = adci_tensor_element_count(tensor);
    const unsigned int output_buffer_size = adci_tensor_element_count(output) * adci_tensor_dtype_size(output->dtype);
    unsigned int volume = 1;
    for(unsigned int i = 0; i < shape->shape[0]; i++){
        output->shape[i] = ((int32_t *)shape->data)[i];
        volume *= output->shape[i];
    }
    assert(volume == required_count);
    output->n_dimension = shape->shape[0];
    if(tensor == output) return;
    /* TO REALLOCATE MEMORY ONLY IN THE CASE OF BUFFER NOT LARGE ENOUGH */
    const unsigned int required_buffer_size = required_count * adci_tensor_dtype_size(tensor->dtype);
    if(output->data != NULL && output_buffer_size < required_buffer_size)
        output->data = ADCI_REALLOC(output->data, required_buffer_size);
    else if(output->data == NULL)
        output->data = ADCI_ALLOC(required_buffer_size);
    output->dtype = tensor->dtype;
    memcpy(output->data, tensor->data, required_buffer_size);
}

void ADCI_EXIT_POINT adci_tensor_copy(struct adci_tensor *input, struct adci_tensor *output){
    if(input == output){
        ADCI_LOG(ADCI_WARNING, "INPUT AND OUTPUT TENSORS FOR COPY OP ARE THE SAME");
        return;
    }
    adci_check_tensor_types((struct adci_tensor *[]){input, output});
    adci_check_tensor_dim((struct adci_tensor *[]){input, output});
    unsigned int volume = 1;
    for(unsigned int i = 0; i < input->n_dimension; i++) volume *= input->shape[i];
    memcpy(output->data, input->data, adci_tensor_dtype_size(input->dtype) * volume);
}

void ADCI_EXIT_POINT adci_tensor_compute_op(struct adci_vector inputs, struct adci_tensor *output, enum adci_tensor_op op){
    switch (op){
    case ADCI_TENSOR_ADD: return adci_tensor_add(inputs, output);
    case ADCI_TENSOR_RESHAPE: return adci_tensor_reshape(inputs, output);
    default:
        assert("TODO, OPERATION NOT IMPLEMENTED YET" == 0);
    }
    assert("SHOULD NEVER REACH" == 0);
}