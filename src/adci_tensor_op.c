#include "adci_common.h"
#include "adci_tensor_op.h"
#include "adci_logging.h"

#include "adci_tensor_common.h"

/* PRIVATE FUNCTIONS */

static void ADCI_EXIT_POINT adci_check_tensor_types(struct adci_tensor **tensors){
    ADCI_ASSERT(tensors[0]->dtype == tensors[1]->dtype);
}

static void ADCI_EXIT_POINT adci_check_tensor_vec_types(struct adci_vector tensors){
    if(tensors.length == 0) return;
    enum adci_tensor_type type = (*(struct adci_tensor **)adci_vector_get(&tensors, 0))->dtype; 
    for(unsigned int i = 1; i < tensors.length; i++){
        enum adci_tensor_type current = (*(struct adci_tensor **)adci_vector_get(&tensors, i))->dtype; 
        ADCI_ASSERT(current == type);
    }
    /* TODO, CONTINUE */
}

/* TODO, ADD SOME KIND OF MACRO TO DISABLE CHECKS TO SPEED UP INFERENCE */
static void ADCI_EXIT_POINT adci_check_tensor_dim(struct adci_tensor **inputs){
    ADCI_ASSERT(inputs[0]->n_dimension == inputs[1]->n_dimension);
    for(unsigned int i = 0; i < inputs[0]->n_dimension; i++){
        ADCI_ASSERT(inputs[0]->shape[i] == inputs[1]->shape[i]);
    }
}

static void adci_reset_value(void *data, enum adci_tensor_type type){
    #define ADCI_RESET(_type) *((_type*)data) = (_type)0
    switch (type){
    case ADCI_F32: ADCI_RESET(float);
    break;
    case ADCI_I32: ADCI_RESET(int32_t);
    break;
    case ADCI_I8: ADCI_RESET(int8_t);
    break;
    default:
        ADCI_LOG(ADCI_ERROR, "RESET FOR TYPE %d, NOT IMPLEMENTED", type);
    }
}

/* TEMPLATE FOR 1 TO 1 OP LIKE ADDITION AND SUBSTRACTION */

#define OP_FOR_TYPE(_first, _output, _type, _op_token) ((_type*)_output->data)[i] = ((_type*)_output->data)[i] _op_token ((_type*)_first->data)[i]
typedef void (*adci_tensor_single_op)(struct adci_vector inputs, struct adci_tensor *output, unsigned int i);
static void ADCI_EXIT_POINT adci_tensor_element_independent_op(struct adci_vector inputs, struct adci_tensor *output, enum adci_tensor_op op, adci_tensor_single_op callback){
    if(inputs.length <= 1){
        ADCI_LOG(ADCI_WARNING, "%s OP WITH ONLY 1 TENSOR INPUT, COPYING TO OUTPUT", adci_tensor_op_str(op));
        adci_tensor_copy(*(struct adci_tensor **)adci_vector_get(&inputs, 0), output);
        return;
    }
    adci_check_tensor_dim(adci_vector_get(&inputs, 0));
    struct adci_tensor *first = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    adci_check_tensor_dim((struct adci_tensor *[]){first, output});
    unsigned int tensor_volume = 1;
    for(unsigned int i = 0; i < first->n_dimension; i++) tensor_volume *= first->shape[i];
    const unsigned int element_size = adci_tensor_dtype_size(output->dtype);
    ADCI_ASSERT(output->data != NULL);
    /* TODO, SPLIT INTO MULTIPLE THREADS */
    for(unsigned int i = 0; i < tensor_volume; i++){
        adci_reset_value((uint8_t *)output->data + i * element_size, output->dtype);
        callback(inputs, output, i);
    }
}   

static void adci_tensor_single_add(struct adci_vector inputs, struct adci_tensor *output, unsigned int i){
    #define ADD_FOR_TYPE(_first, _output, _type) OP_FOR_TYPE(_first, _output, _type, +)
    for(unsigned int j = 0; j < inputs.length; j++){
        struct adci_tensor *current = *(struct adci_tensor **)adci_vector_get(&inputs, j);
        ADCI_ASSERT(current->data != NULL);
        switch (current->dtype){
        case ADCI_F32: ADD_FOR_TYPE(current, output, float);
        break;
        case ADCI_I32: ADD_FOR_TYPE(current, output, int32_t);
        break;
        case ADCI_I8: ADD_FOR_TYPE(current, output, int8_t);
        break;
        default: ADCI_LOG(ADCI_ERROR, "ADD FOR TYPE %d, NOT IMPLEMENTED", current->dtype);
        break;
        }
    }
}

static void adci_tensor_single_sub(struct adci_vector inputs, struct adci_tensor *output, unsigned int i){
    #define SUB_FOR_TYPE(_first, _output, _type) OP_FOR_TYPE(_first, _output, _type, -)
    for(unsigned int j = 0; j < inputs.length; j++){
        struct adci_tensor *current = *(struct adci_tensor **)adci_vector_get(&inputs, j);
        ADCI_ASSERT(current->data != NULL);
        if(j == 0){
            const unsigned int offset = i * adci_tensor_dtype_size(current->dtype);
            memcpy((uint8_t *)output->data + offset, (uint8_t *)current->data + offset, adci_tensor_dtype_size(current->dtype));
            continue;
        } 
        switch (current->dtype){
        case ADCI_F32: SUB_FOR_TYPE(current, output, float);
        break;
        case ADCI_I32: SUB_FOR_TYPE(current, output, int32_t);
        break;
        case ADCI_I8: SUB_FOR_TYPE(current, output, int8_t);
        break;
        default: ADCI_LOG(ADCI_ERROR, "SUB FOR TYPE %d, NOT IMPLEMENTED", current->dtype);
        break;
        }
    }
}

typedef void (*single_op_template_fn_t)(const void *first, const void *second, void *output);
#define IMPLEMENT_FUNCTIONS_FOR_OP_TEMPLATE(_TEMPLATE_FN) \
    _TEMPLATE_FN(float, float) \
    _TEMPLATE_FN(float, int32_t) \
    _TEMPLATE_FN(float, int8_t) \
    _TEMPLATE_FN(int32_t, float) \
    _TEMPLATE_FN(int32_t, int32_t) \
    _TEMPLATE_FN(int32_t, int8_t) \
    _TEMPLATE_FN(int8_t, float) \
    _TEMPLATE_FN(int8_t, int32_t) \
    _TEMPLATE_FN(int8_t, int8_t)

#define INIT_FUNCTION_LIST_FOR_OP_TEMPLATE(_list_name, _GET_TEMPLATE_FN) \
static single_op_template_fn_t _list_name[ADCI_NUM_SUPPORTED_TYPES * ADCI_NUM_SUPPORTED_TYPES] = { \
    _GET_TEMPLATE_FN(float, float), \
    _GET_TEMPLATE_FN(float, int32_t), \
    _GET_TEMPLATE_FN(float, int8_t), \
    _GET_TEMPLATE_FN(int32_t, float), \
    _GET_TEMPLATE_FN(int32_t, int32_t), \
    _GET_TEMPLATE_FN(int32_t, int8_t), \
    _GET_TEMPLATE_FN(int8_t, float), \
    _GET_TEMPLATE_FN(int8_t, int32_t), \
    _GET_TEMPLATE_FN(int8_t, int8_t) \
};

#define SINGLE_PRELU_FN_NAME(_ftype, _stype) single_prelu_op_template_fn_ ## _ftype ## _ ## _stype
#define SINGLE_PRELU_OP_TEMPLATE_FN(_ftype, _stype)  \
static void SINGLE_PRELU_FN_NAME(_ftype, _stype)(const void *first, const void *second, void *output){ \
    ((_ftype *)output)[0] = ((_ftype *)first)[0]; \
    if(((_ftype *)first)[0] > 0) return; \
    ((_ftype *)output)[0] = (_ftype) (((_ftype *)first)[0] * ((_stype *)second)[0]);    \
}
IMPLEMENT_FUNCTIONS_FOR_OP_TEMPLATE(SINGLE_PRELU_OP_TEMPLATE_FN)
INIT_FUNCTION_LIST_FOR_OP_TEMPLATE(single_prelu_op_template_fns, SINGLE_PRELU_FN_NAME)

/* END PRIVATE FUNCTIONS */

void ADCI_EXIT_POINT adci_tensor_add(struct adci_vector inputs, struct adci_tensor *output){
    adci_tensor_element_independent_op(inputs, output, ADCI_TENSOR_ADD, adci_tensor_single_add);
}

void ADCI_EXIT_POINT adci_tensor_sub(struct adci_vector inputs, struct adci_tensor *output){
    adci_tensor_element_independent_op(inputs, output, ADCI_TENSOR_SUB, adci_tensor_single_sub);
}

/* FIRST ELEMENT IS THE TENSOR TO RESHAPE AND SECOND IS THE SHAPE TENSOR (DIM IS ONE)*/
void ADCI_EXIT_POINT adci_tensor_reshape(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 2);
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    struct adci_tensor *shape = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    ADCI_ASSERT(shape->n_dimension == 1);
    ADCI_ASSERT(shape->data != NULL);
    const unsigned int required_count = adci_tensor_element_count(tensor);
    const unsigned int output_buffer_size = adci_tensor_element_count(output) * adci_tensor_dtype_size(output->dtype);
    unsigned int volume = 1;
    for(unsigned int i = 0; i < shape->shape[0]; i++){
        output->shape[i] = ((int32_t *)shape->data)[i];
        volume *= output->shape[i];
    }
    ADCI_ASSERT(volume == required_count);
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

static void adci_tensor_pad_fill_data(const struct adci_tensor *input, const struct adci_tensor *padding, struct adci_tensor *output, unsigned int dim, unsigned int in_offset, unsigned int ou_offset){
    const unsigned int bsize = adci_tensor_dtype_size(input->dtype);
    unsigned int ou_volume = adci_tensor_element_count_ext(output->n_dimension - dim - 1, output->shape + dim + 1); 
    unsigned int in_volume = adci_tensor_element_count_ext(input->n_dimension - dim - 1, input->shape + dim + 1);
    for(unsigned int i = 0; i < input->shape[dim]; i++){
        const unsigned int curr_ou_offset = (i + ((int32_t *)padding->data)[dim * padding->shape[1]]) * ou_volume;
        const unsigned int curr_in_offset = i * in_volume;
        if(dim == input->n_dimension - 1){
            const unsigned int fou_offset = (ou_offset + curr_ou_offset) * bsize;
            const unsigned int fin_offset = (in_offset + curr_in_offset) * bsize;
            memcpy((int8_t *)output->data + fou_offset, (int8_t *)input->data + fin_offset, bsize);
        }else adci_tensor_pad_fill_data(input, padding, output, dim + 1, in_offset + curr_in_offset, ou_offset + curr_ou_offset);
    }
}

void ADCI_EXIT_POINT adci_tensor_pad(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 2);
    struct adci_tensor *element = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    struct adci_tensor *padding = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    ADCI_ASSERT(padding->dtype == ADCI_I32);
    ADCI_ASSERT(padding->n_dimension == 2);
    ADCI_ASSERT(element->n_dimension == padding->shape[0]);
    const unsigned int previous_count = adci_tensor_element_count(element);
    unsigned int volume = 1;
    for(unsigned int i = 0; i < padding->shape[0]; i++){
        const unsigned int pad_size = ((int32_t *)padding->data)[i * padding->shape[1]] + ((int32_t *)padding->data)[i * padding->shape[1] + 1];
        output->shape[i] = element->shape[i] + pad_size;
        volume *= element->shape[i] + pad_size;
    }
    if(volume == previous_count){
        if(element != output) adci_tensor_copy(element, output);
        return;
    }
    /* INCREASE SIZE OF TENSOR */
    const unsigned int padded_tensor_size = volume * adci_tensor_dtype_size(element->dtype);
    if(output->data) ADCI_FREE(output->data);
    output->data = ADCI_ALLOC(padded_tensor_size);
    output->dtype = element->dtype;
    output->n_dimension = element->n_dimension;
    memset(output->data, 0, padded_tensor_size);
    adci_tensor_pad_fill_data(element, padding, output, 0, 0, 0);
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

void ADCI_EXIT_POINT adci_tensor_prelu(struct adci_vector inputs, struct adci_tensor *output){
    /* f(y) = y if y >= 0 and f(y) = a * y if y < 0 */
    /* EACH CHANNEL HAS A DIFFERENT PARAMETER a */
    struct adci_tensor *element = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    struct adci_tensor *parameters = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    ADCI_ASSERT(element->dtype == output->dtype);
    adci_check_tensor_dim((struct adci_tensor *[]){element, output});
    ADCI_ASSERT(parameters->n_dimension <= element->n_dimension);
    for(unsigned int i = 0; i < parameters->n_dimension; i++){
        ADCI_ASSERT(
            element->shape[element->n_dimension - i - 1] == 1        || 
            parameters->shape[parameters->n_dimension - i - 1] == 1  || 
            element->shape[element->n_dimension - i - 1] == parameters->shape[parameters->n_dimension - i - 1]);
    }
    const unsigned int volume = adci_tensor_element_count(element);
    const unsigned int param_volume = adci_tensor_element_count(parameters);
    single_op_template_fn_t prelu = single_prelu_op_template_fns[element->dtype * ADCI_NUM_SUPPORTED_TYPES + parameters->dtype];
    const unsigned int input_elem_size = adci_tensor_dtype_size(element->dtype);
    const unsigned int param_elem_size = adci_tensor_dtype_size(parameters->dtype);
    for(unsigned int i = 0; i < volume; i++){
        const unsigned int offset = i * input_elem_size;
        const unsigned int param_offset = (i % param_volume) * param_elem_size; 
        prelu(element->data + offset, parameters->data + param_offset, output->data + offset);
    }
}

void ADCI_EXIT_POINT adci_tensor_compute_op(struct adci_vector inputs, struct adci_tensor *output, enum adci_tensor_op op){
    switch (op){
    case ADCI_TENSOR_INPUT: return;
    case ADCI_TENSOR_ADD: return adci_tensor_add(inputs, output);
    case ADCI_TENSOR_SUB: return adci_tensor_sub(inputs, output);
    case ADCI_TENSOR_RESHAPE: return adci_tensor_reshape(inputs, output);
    case ADCI_TENSOR_COPY: return adci_tensor_copy(*(struct adci_tensor**)adci_vector_get(&inputs, 0), output);
    case ADCI_TENSOR_PAD: return adci_tensor_pad(inputs, output);
    default:
        ADCI_LOG(ADCI_ERROR, "OPERATION: %s NOT IMPLEMENTED YET", adci_tensor_op_str(op));
        ADCI_ASSERT(false);
    }
}

const char * adci_tensor_op_str(enum adci_tensor_op op){
    #define OP_STR_CASE(_op) case _op: return ADCI_TOKEN2STR(_op)
    switch (op){
        OP_STR_CASE(ADCI_TENSOR_COPY);
        OP_STR_CASE(ADCI_TENSOR_ADD);
        OP_STR_CASE(ADCI_TENSOR_SUB);
        OP_STR_CASE(ADCI_TENSOR_MUL);
        OP_STR_CASE(ADCI_TENSOR_BATCH_MATMUL);
        OP_STR_CASE(ADCI_TENSOR_PAD);
        OP_STR_CASE(ADCI_TENSOR_CONV2D);
        OP_STR_CASE(ADCI_TENSOR_PRELU);
        OP_STR_CASE(ADCI_TENSOR_CONCAT);
        OP_STR_CASE(ADCI_TENSOR_AVG_POOL2D);
        OP_STR_CASE(ADCI_TENSOR_TRANSPOSE);
        OP_STR_CASE(ADCI_TENSOR_RESHAPE);
        OP_STR_CASE(ADCI_TENSOR_REDUCE_MAX);
        OP_STR_CASE(ADCI_TENSOR_SOFTMAX);
        OP_STR_CASE(ADCI_TENSOR_TRANSPOSE_CONV);
        OP_STR_CASE(ADCI_TENSOR_INPUT);
        default: return "INVALID OP";
    }
}
    