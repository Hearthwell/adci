#include <math.h>

#include "adci_common.h"
#include "adci_tensor_op.h"
#include "adci_logging.h"

#include "adci_tensor_common.h"

/* PRIVATE FUNCTIONS */

static void ADCI_EXIT_POINT adci_check_tensor_dim(const struct adci_tensor **inputs){
    ADCI_ASSERT(inputs[0]->n_dimension == inputs[1]->n_dimension);
    for(unsigned int i = 0; i < inputs[0]->n_dimension; i++){
        ADCI_ASSERT(inputs[0]->shape[i] == inputs[1]->shape[i]);
    }
}

static void adci_reset_value(void *data, enum adci_tensor_type type){
    #define ADCI_RESET(_type) *((_type*)data) = (_type)0
    switch (type){
    case ADCI_F32: ADCI_RESET(float); break;
    case ADCI_I32: ADCI_RESET(int32_t); break;
    case ADCI_I8: ADCI_RESET(int8_t); break;
    default:
        ADCI_LOG(ADCI_ERROR, "RESET FOR TYPE %d, NOT IMPLEMENTED", type);
    }
}

static void adci_compare_max(const void *data, void *maximum, enum adci_tensor_type type){
    #define ADCI_CMP_MAX(_type) *((_type*)maximum) = (*((_type*)maximum) < *((_type*)data)) ? *((_type*)data) : *((_type*)maximum)
    switch (type){
    case ADCI_F32: ADCI_CMP_MAX(float); break;
    case ADCI_I32: ADCI_CMP_MAX(int32_t); break;
    case ADCI_I8: ADCI_CMP_MAX(int8_t); break;
    default:
        ADCI_LOG(ADCI_ERROR, "RESET FOR TYPE %d, NOT IMPLEMENTED", type);
    }
}

static unsigned int adci_prepare_output_tensor(unsigned int *shape, unsigned int length, enum adci_tensor_type type, struct adci_tensor *tensor){
    const unsigned int previous_size = adci_tensor_element_count(tensor) * adci_tensor_dtype_size(tensor->dtype);
    const unsigned int required_size = adci_tensor_element_count_ext(length, shape) * adci_tensor_dtype_size(type);
    tensor->n_dimension = length;
    tensor->dtype = type;
    memcpy(tensor->shape, shape, length * sizeof(unsigned int));
    if(tensor->data == NULL) tensor->data = ADCI_ALLOC(required_size);
    /* TO REALLOCATE MEMORY ONLY IN THE CASE OF BUFFER NOT LARGE ENOUGH */
    else if(required_size > previous_size) tensor->data = ADCI_REALLOC(tensor->data, required_size);
    return required_size;
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
    adci_prepare_output_tensor(first->shape, first->n_dimension, first->dtype, output);
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

/* TODO, CHANGE TO USE NEW MECANISM TO SUPPORT MIXING TYPES */
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

/* TODO, CHANGE TO USE NEW MECANISM TO SUPPORT MIXING TYPES */
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

static void adci_reduce_max_next(const struct adci_tensor *tensor, unsigned int *indeces, const struct adci_vector dim_mapping){
    for(unsigned int i = 0; i < dim_mapping.length; i++){
        const unsigned int index = dim_mapping.length - 1 - i;
        indeces[index] += 1;
        if(indeces[index] < tensor->shape[((unsigned int *)dim_mapping.data)[index]]) break;
        indeces[index] = 0;
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

/* PRELU FUNCTIONS IMPLEMENTATION */
#define SINGLE_PRELU_FN_NAME(_ftype, _stype) single_prelu_op_template_fn_ ## _ftype ## _ ## _stype
#define SINGLE_PRELU_OP_TEMPLATE_FN(_ftype, _stype)  \
static void SINGLE_PRELU_FN_NAME(_ftype, _stype)(const void *first, const void *second, void *output){ \
    ((_ftype *)output)[0] = ((_ftype *)first)[0]; \
    if(((_ftype *)first)[0] > 0) return; \
    ((_ftype *)output)[0] = (_ftype) (((_ftype *)first)[0] * ((_stype *)second)[0]);    \
}
IMPLEMENT_FUNCTIONS_FOR_OP_TEMPLATE(SINGLE_PRELU_OP_TEMPLATE_FN)
INIT_FUNCTION_LIST_FOR_OP_TEMPLATE(single_prelu_op_template_fns, SINGLE_PRELU_FN_NAME)

/* CAST FUNCTIONS IMPLEMENTATION _ftype: initial type, _stype: type to be casted to */
#define SINGLE_CAST_FN_NAME(_ftype, _stype) single_cast_op_template_fn_ ## _ftype ## _ ## _stype
#define SINGLE_CAST_OP_TEMPLATE_FN(_ftype, _stype)  \
static void SINGLE_CAST_FN_NAME(_ftype, _stype)(const void *first, const void *second, void *output){ \
    (void)second; \
    ((_stype *)output)[0] = (_stype)((_ftype *)first)[0]; \
}
IMPLEMENT_FUNCTIONS_FOR_OP_TEMPLATE(SINGLE_CAST_OP_TEMPLATE_FN)
INIT_FUNCTION_LIST_FOR_OP_TEMPLATE(single_cast_op_template_fns, SINGLE_CAST_FN_NAME)

/* MULTIPLY FUNCTIONS IMPLEMENTATION */
#define SINGLE_MULT_FN_NAME(_ftype, _stype) single_mult_op_template_fn_ ## _ftype ## _ ## _stype
#define SINGLE_MULT_OP_TEMPLATE_FN(_ftype, _stype)  \
static void SINGLE_MULT_FN_NAME(_ftype, _stype)(const void *first, const void *second, void *output){ \
    ((_ftype *)output)[0] = (_ftype)(((_ftype *)first)[0] * ((_stype *)second)[0]); \
}
IMPLEMENT_FUNCTIONS_FOR_OP_TEMPLATE(SINGLE_MULT_OP_TEMPLATE_FN)
INIT_FUNCTION_LIST_FOR_OP_TEMPLATE(single_mult_op_template_fns, SINGLE_MULT_FN_NAME)

/* END PRIVATE FUNCTIONS */

void ADCI_EXIT_POINT adci_tensor_add(struct adci_vector inputs, struct adci_tensor *output){
    adci_tensor_element_independent_op(inputs, output, ADCI_TENSOR_ADD, adci_tensor_single_add);
}

void ADCI_EXIT_POINT adci_tensor_sub(struct adci_vector inputs, struct adci_tensor *output){
    adci_tensor_element_independent_op(inputs, output, ADCI_TENSOR_SUB, adci_tensor_single_sub);
}

/* FIRST ELEMENT IS THE TENSOR TO RESHAPE AND SECOND IS THE SHAPE TENSOR (DIM IS ONE) */
void ADCI_EXIT_POINT adci_tensor_reshape(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 2);
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    struct adci_tensor *shape = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    ADCI_ASSERT(shape->n_dimension == 1);
    ADCI_ASSERT(shape->data != NULL);
    ADCI_ASSERT(shape->dtype == ADCI_I32);
    const unsigned int tensor_size = adci_prepare_output_tensor((unsigned int *)shape->data, shape->shape[0], tensor->dtype, output);
    if(tensor == output) return;
    memcpy(output->data, tensor->data, tensor_size);
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
    unsigned int output_shape[element->n_dimension];
    for(unsigned int i = 0; i < padding->shape[0]; i++){
        const unsigned int pad_size = ((int32_t *)padding->data)[i * padding->shape[1]] + ((int32_t *)padding->data)[i * padding->shape[1] + 1];
        output_shape[i] = element->shape[i] + pad_size;
        volume *= element->shape[i] + pad_size;
    }
    struct adci_tensor temp = *element;
    /* SO WE DONT CLEAR INPUT TENSOR BEFORE MOVING THE DATA */
    if(element == output) output->data = NULL;
    adci_prepare_output_tensor(output_shape, element->n_dimension, element->dtype, output);
    if(volume == previous_count){
        if(element != output) adci_tensor_copy(element, output);
        return;
    }
    const unsigned int padded_tensor_size = volume * adci_tensor_dtype_size(element->dtype);
    memset(output->data, 0, padded_tensor_size);
    adci_tensor_pad_fill_data(&temp, padding, output, 0, 0, 0);
    if(element == output) ADCI_FREE(temp.data);
}

void ADCI_EXIT_POINT adci_tensor_copy(struct adci_tensor *input, struct adci_tensor *output){
    if(input == output){
        ADCI_LOG(ADCI_WARNING, "INPUT AND OUTPUT TENSORS FOR COPY OP ARE THE SAME");
        return;
    }
    unsigned int output_size = adci_prepare_output_tensor(input->shape, input->n_dimension, input->dtype, output);
    memcpy(output->data, input->data, output_size);
}

void ADCI_EXIT_POINT adci_tensor_prelu(struct adci_vector inputs, struct adci_tensor *output){
    /* f(y) = y if y >= 0 and f(y) = a * y if y < 0 */
    /* EACH CHANNEL HAS A DIFFERENT PARAMETER a */
    struct adci_tensor *element = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    struct adci_tensor *parameters = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    ADCI_ASSERT(parameters->n_dimension <= element->n_dimension);
    for(unsigned int i = 0; i < parameters->n_dimension; i++){
        ADCI_ASSERT(
            element->shape[element->n_dimension - i - 1] == 1        || 
            parameters->shape[parameters->n_dimension - i - 1] == 1  || 
            element->shape[element->n_dimension - i - 1] == parameters->shape[parameters->n_dimension - i - 1]);
    }
    struct adci_tensor temp = *element;
    if(element == output) output->data = NULL;
    const unsigned int output_size = adci_prepare_output_tensor(element->shape, element->n_dimension, element->dtype, output);
    const unsigned int input_elem_size = adci_tensor_dtype_size(element->dtype);
    const unsigned int param_elem_size = adci_tensor_dtype_size(parameters->dtype);
    const unsigned int volume = output_size / input_elem_size;
    const unsigned int param_volume = adci_tensor_element_count(parameters);
    single_op_template_fn_t prelu = single_prelu_op_template_fns[element->dtype * ADCI_NUM_SUPPORTED_TYPES + parameters->dtype];
    for(unsigned int i = 0; i < volume; i++){
        const unsigned int offset = i * input_elem_size;
        const unsigned int param_offset = (i % param_volume) * param_elem_size; 
        prelu(temp.data + offset, parameters->data + param_offset, output->data + offset);
    }
    if(element == output) ADCI_FREE(temp.data);
}

void ADCI_EXIT_POINT adci_tensor_cast(struct adci_vector inputs, struct adci_tensor *output){
    struct adci_tensor *input = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    if(input == output){
        ADCI_LOG(ADCI_WARNING, "SAME INPUT,OUTPUT TENSOR FOR CAST OPERATIONS");
        return;
    } 
    const unsigned int in_elem_size = adci_tensor_dtype_size(input->dtype);
    const unsigned int ou_elem_size = adci_tensor_dtype_size(output->dtype);
    /* TODO, CHANGE, SHOULD PROBABLY GET OUTPUT TYPE FROM AN INPUT TENSOR */
    const unsigned int output_size = adci_prepare_output_tensor(input->shape, input->n_dimension, output->dtype, output);
    unsigned int volume = output_size / ou_elem_size;
    const single_op_template_fn_t cast = single_cast_op_template_fns[input->dtype * ADCI_NUM_SUPPORTED_TYPES + output->dtype]; 
    for(unsigned int i = 0; i < volume; i++)
        cast(input->data + i * in_elem_size, NULL, output->data + i * ou_elem_size);
}

void ADCI_EXIT_POINT adci_tensor_softmax(struct adci_vector inputs, struct adci_tensor *output){
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    struct adci_tensor *axis = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    ADCI_ASSERT(axis->n_dimension == 1 && axis->shape[0] == 1);
    ADCI_ASSERT(axis->dtype == ADCI_I32);
    /* TODO, ADD SUPPORT FOR F64, F16 */
    ADCI_ASSERT(output->dtype == ADCI_F32);
    const unsigned int dim = adci_tensor_get_i32(axis, 0);
    ADCI_ASSERT(dim < tensor->n_dimension);
    struct adci_tensor temp = *tensor;
    if(tensor == output) output->data = NULL;
    adci_prepare_output_tensor(tensor->shape, tensor->n_dimension, output->dtype, output);
    const unsigned int count = adci_tensor_element_count(&temp) / temp.shape[dim];
    const unsigned int in_elem_size = adci_tensor_dtype_size(temp.dtype);
    const single_op_template_fn_t cast = single_cast_op_template_fns[temp.dtype * ADCI_NUM_SUPPORTED_TYPES + output->dtype]; 
    /* WILL CONTAIN THE TEMPORARY INPUT CASTED VALUES */
    float elements[temp.shape[dim]];
    for(unsigned int i = 0; i < count; i++){
        float sum = 0;
        for(unsigned int j = 0; j < temp.shape[dim]; j++){
            /* NO NEED FOR THIS STEP IF INPUT'S TYPE IS F32 */
            cast(temp.data + (j * count + i) * in_elem_size, NULL, elements + j);
            sum += exp(elements[j]);
        }
        for(unsigned int j = 0; j < temp.shape[dim]; j++)
            ((float *)output->data)[j * count + i] = exp(elements[j]) / sum;
    }
    if(tensor == output) ADCI_FREE(temp.data);
}

void ADCI_EXIT_POINT adci_tensor_reduce_max(struct adci_vector inputs, struct adci_tensor *output){
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    struct adci_tensor *axis = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    ADCI_ASSERT(axis->n_dimension == 1);
    ADCI_ASSERT(axis->dtype == ADCI_I32);
    ADCI_ASSERT(axis->shape[0] <= tensor->n_dimension);
    bool keep_dims = false;
    if(inputs.length >= 3){
        struct adci_tensor *format = *(struct adci_tensor **)adci_vector_get(&inputs, 2);
        ADCI_ASSERT(format->dtype == ADCI_I32);
        ADCI_ASSERT(format->n_dimension == 1);
        keep_dims = adci_tensor_get_i32(format, 0) != 0;
    }
    const unsigned int element_size = adci_tensor_dtype_size(tensor->dtype);
    struct adci_tensor temp = *tensor;
    if(tensor == output) output->data = NULL;
    memcpy(output->shape, temp.shape, sizeof(temp.shape));
    unsigned int reduce_volume = 1;
    struct adci_vector free_dim_mapping = adci_vector_init(sizeof(unsigned int));
    struct adci_vector reduced_dim_mapping = adci_vector_init(sizeof(unsigned int));
    struct adci_vector axis_view = {.data = axis->data, .bsize = adci_tensor_dtype_size(axis->dtype), .length = axis->shape[0]};
    for(unsigned int i = 0; i < temp.n_dimension; i++){
        if(!adci_vector_has(&axis_view, &i)){
            adci_vector_add(&free_dim_mapping, &i);
            continue;
        }
        /* PART OF THE REDUCED DIMS */
        adci_vector_add(&reduced_dim_mapping, &i);
        reduce_volume *= temp.shape[i];
    }
    /* GIVE FINAL SIZE TO OUTPUT TENSOR */
    for(unsigned int i = 0; i < free_dim_mapping.length; i++) 
        output->shape[i] = temp.shape[((unsigned int *)free_dim_mapping.data)[i]];
    const unsigned int output_size = adci_prepare_output_tensor(output->shape, free_dim_mapping.length, temp.dtype, output);
    const unsigned int out_volume = output_size / element_size;
    if(output->n_dimension == 0){
        output->n_dimension = 1;
        output->shape[0] = 1;
    }
    unsigned int dim_volumes[temp.n_dimension];
    for(unsigned int i = 0; i < temp.n_dimension; i++)
        dim_volumes[i] = adci_tensor_element_count_ext(temp.n_dimension - i - 1, temp.shape + i + 1);
    int8_t maximum[element_size];
    unsigned int free_dim_index[temp.n_dimension - axis->shape[0]];
    unsigned int reduced_dim_index[axis->shape[0]];
    memset(free_dim_index, 0, sizeof(free_dim_index));
    for(unsigned int i = 0; i < out_volume; i++){
        unsigned int fixed_offset = 0;
        for(unsigned int j = 0; j < free_dim_mapping.length; j++)
            fixed_offset += free_dim_index[j] * dim_volumes[((unsigned int *)free_dim_mapping.data)[j]];
        adci_reset_value(maximum, tensor->dtype);
        memset(reduced_dim_index, 0, sizeof(reduced_dim_index));
        for(unsigned int j = 0; j < reduce_volume; j++){
            /* CHECK FOR MAXIMUM VALUE AMONGST THE REDUCED DIMENSIONS */
            unsigned int reduced_offset = 0;
            for(unsigned int k = 0; k < reduced_dim_mapping.length; k++)
                reduced_offset += reduced_dim_index[k] * dim_volumes[((unsigned int *)reduced_dim_mapping.data)[k]];
            int8_t *current = (int8_t *)temp.data + (fixed_offset + reduced_offset) * element_size;
            adci_compare_max(current, maximum, temp.dtype);
            adci_reduce_max_next(&temp, reduced_dim_index, reduced_dim_mapping);
        }
        memcpy((int8_t *)output->data + i * element_size, maximum, element_size);
        adci_reduce_max_next(&temp, free_dim_index, free_dim_mapping);
    }
    if(keep_dims){
        output->n_dimension = temp.n_dimension;
        for(unsigned int i = 0; i < temp.n_dimension; i++){
            output->shape[i] = temp.shape[i];
            if(adci_vector_has(&axis_view, &i)) output->shape[i] = 1; 
        }
    }
    adci_vector_free(&reduced_dim_mapping);
    adci_vector_free(&free_dim_mapping);
    if(tensor == output) ADCI_FREE(temp.data);
}

void ADCI_EXIT_POINT adci_tensor_concat(struct adci_vector inputs, struct adci_tensor *output){
    const struct adci_tensor *axis = *(struct adci_tensor **)adci_vector_get(&inputs, inputs.length - 1); 
    ADCI_ASSERT(axis->n_dimension == 1);
    ADCI_ASSERT(axis->dtype == ADCI_I32);
    const unsigned int dim = adci_tensor_get_i32(axis, 0);
    struct adci_tensor *first = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    ADCI_ASSERT(dim < first->n_dimension);
    unsigned int concat_dim_size = first->shape[dim];
    bool output_in_input = first == output;
    for(unsigned int i = 1; i < inputs.length - 1; i++){
        /* MAKE SURE ALL TENSORS TO BE CONCATENATED HAVE SAME SHAPE EXCEPT FOR THE DIM TO BE CONCATENATED */
        struct adci_tensor *current = *(struct adci_tensor **)adci_vector_get(&inputs, i);
        ADCI_ASSERT(first->n_dimension == current->n_dimension);
        ADCI_ASSERT(first->dtype == current->dtype);
        if(current == output) output_in_input = true;
        for(unsigned int j = 0; j < current->n_dimension; j++){
            if(j == dim){
                concat_dim_size += current->shape[j];
                continue;
            } 
            ADCI_ASSERT(current->shape[j] == first->shape[j]);
        }
    }
    /* ALL THE DIMS MATCH EXCEPT FOR THE CONCATENATED DIM */
    unsigned int count = adci_tensor_element_count_ext(dim, first->shape);
    struct adci_tensor temp = *output;
    if(output_in_input) output->data = NULL;
    unsigned int curr_shape[first->n_dimension];
    memcpy(curr_shape, first->shape, first->n_dimension * sizeof(unsigned int));
    curr_shape[dim] = concat_dim_size;
    adci_prepare_output_tensor(curr_shape, first->n_dimension, first->dtype, output);
    const unsigned int element_size = adci_tensor_dtype_size(first->dtype);
    const unsigned int copy_block_size = adci_tensor_element_count_ext(first->n_dimension - dim - 1, first->shape + dim + 1) * element_size;
    const unsigned int output_single_copy_block_size = concat_dim_size * copy_block_size;
    for(unsigned int i = 0; i < count; i++){
        unsigned int offset = 0;
        for(unsigned int j = 0; j < inputs.length - 1; j++){
            const struct adci_tensor *current = *(struct adci_tensor **)adci_vector_get(&inputs, j);
            const unsigned int curr_copy_size = current->shape[dim] * copy_block_size;
            memcpy((int8_t *)output->data + i * output_single_copy_block_size + offset, 
                (int8_t *)current->data + i * curr_copy_size, curr_copy_size);
            offset += curr_copy_size;
        }
    }
    if(output_in_input) ADCI_FREE(temp.data);
}

void ADCI_EXIT_POINT adci_tensor_mul(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 2);
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    struct adci_tensor *mult = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    for(unsigned int i = 0; i < mult->n_dimension; i++){
        ADCI_ASSERT(
            tensor->shape[tensor->n_dimension - i - 1] == 1 ||
            mult->shape[mult->n_dimension - i - 1] == 1 ||
            tensor->shape[tensor->n_dimension - i - 1] == mult->shape[mult->n_dimension - i - 1]
        );
    }
    /* BROADCASTING VALID */
    struct adci_tensor temp = *tensor;
    if(tensor == output) output->data = NULL;
    adci_prepare_output_tensor(tensor->shape, tensor->n_dimension, tensor->dtype, output);
    const unsigned int volume = adci_tensor_element_count(tensor);
    const unsigned int mult_volume = adci_tensor_element_count(mult);
    const unsigned int element_size = adci_tensor_dtype_size(tensor->dtype);
    const unsigned int mult_element_size = adci_tensor_dtype_size(mult->dtype);
    single_op_template_fn_t single_mult = single_mult_op_template_fns[tensor->dtype * ADCI_NUM_SUPPORTED_TYPES + mult->dtype];
    for(unsigned int i = 0; i < volume; i++)
        single_mult(tensor->data + i * element_size, mult->data + (i % mult_volume) * mult_element_size, output->data + i * element_size);
    if(tensor == output) ADCI_FREE(temp.data);
}

void ADCI_EXIT_POINT adci_tensor_compute_op(struct adci_vector inputs, struct adci_tensor *output, enum adci_tensor_op op){
    switch (op){
    case ADCI_TENSOR_INPUT: return;
    case ADCI_TENSOR_ADD: return adci_tensor_add(inputs, output);
    case ADCI_TENSOR_SUB: return adci_tensor_sub(inputs, output);
    case ADCI_TENSOR_RESHAPE: return adci_tensor_reshape(inputs, output);
    case ADCI_TENSOR_COPY: return adci_tensor_copy(*(struct adci_tensor**)adci_vector_get(&inputs, 0), output);
    case ADCI_TENSOR_PAD: return adci_tensor_pad(inputs, output);
    case ADCI_TENSOR_PRELU: return adci_tensor_prelu(inputs, output);
    case ADCI_TENSOR_CAST: return adci_tensor_cast(inputs, output);
    case ADCI_TENSOR_SOFTMAX: return adci_tensor_softmax(inputs, output);
    case ADCI_TENSOR_REDUCE_MAX: return adci_tensor_reduce_max(inputs, output);
    case ADCI_TENSOR_CONCAT: return adci_tensor_concat(inputs, output);
    case ADCI_TENSOR_MUL: return adci_tensor_mul(inputs, output);
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
    