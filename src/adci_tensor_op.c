#include "adci_tensor_op.h"

/* PRIVATE FUNCTIONS */

/* TODO, ADD SOME KIND OF MACRO TO DISABLE CHECKS TO SPEED UP INFERENCE */
static void adci_check_tensor_dim(struct adci_tensor **inputs){
    assert(inputs[0]->n_dimension == inputs[1]->n_dimension);
    for(unsigned int i = 0; i < inputs[0]->n_dimension; i++){
        assert(inputs[0]->shape[i] == inputs[1]->shape[i]);
    }
}

/* END PRIVATE FUNCTIONS */

void adci_tensor_add(struct adci_tensor **inputs, struct adci_tensor *output){
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
}

void adci_tensor_reshape(struct adci_tensor *input, unsigned int *shape, unsigned int n_dims){
    const unsigned int required_count = adci_tensor_element_count(input);
    struct adci_tensor reshaped = {.n_dimension = n_dims};
    memcpy(reshaped.shape, shape, n_dims * sizeof(reshaped.shape[0]));
    const unsigned int reshape_count = adci_tensor_element_count(&reshaped);
    assert(required_count == reshape_count);
    for(unsigned int i = 0; i < n_dims; i++)
        input->shape[i] = shape[i];
    input->n_dimension = n_dims;
}

void adci_tensor_compute_op(struct adci_tensor **inputs, struct adci_tensor *output, enum adci_tensor_op op){
    switch (op){
    case ADCI_TENSOR_ADD: return adci_tensor_add(inputs, output);
    case ADCI_TENSOR_RESHAPE: return adci_tensor_reshape(inputs[0], inputs[1]->shape, adci_tensor_element_count(inputs[1]));
    default:
        assert("TODO, OPERATION NOT IMPLEMENTED YET" == 0);
    }
    assert("SHOULD NEVER REACH" == 0);
}