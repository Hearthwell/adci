#include <math.h>

#include "adci_common.h"
#include "adci_tensor_op.h"
#include "adci_logging.h"

#include "adci_tensor_common.h"


/* PRIVATE FUNCTIONS */

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

/* MAX FUNCTIONS IMPLEMENTATION */
#define SINGLE_MAX_FN_NAME(_ftype, _stype) single_max_op_template_fn_ ## _ftype ## _ ## _stype
#define SINGLE_MAX_OP_TEMPLATE_FN(_ftype, _stype)  \
static void SINGLE_MAX_FN_NAME(_ftype, _stype)(const void *first, const void *second, void *output){ \
    ((_ftype *)output)[0] = ((_ftype)(((_ftype *)first)[0] > ((_stype *)second)[0])) ? ((_ftype *)first)[0] : ((_stype *)second)[0]; \
}
IMPLEMENT_FUNCTIONS_FOR_OP_TEMPLATE(SINGLE_MAX_OP_TEMPLATE_FN)
INIT_FUNCTION_LIST_FOR_OP_TEMPLATE(single_max_op_template_fns, SINGLE_MAX_FN_NAME)

/* ADD FUNCTIONS IMPLEMENTATION */
#define SINGLE_ADD_FN_NAME(_ftype, _stype) single_add_op_template_fn_ ## _ftype ## _ ## _stype
#define SINGLE_ADD_OP_TEMPLATE_FN(_ftype, _stype)  \
static void SINGLE_ADD_FN_NAME(_ftype, _stype)(const void *first, const void *second, void *output){ \
    ((_ftype *)output)[0] = (_ftype)(((_ftype *)first)[0] + ((_stype *)second)[0]); \
}
IMPLEMENT_FUNCTIONS_FOR_OP_TEMPLATE(SINGLE_ADD_OP_TEMPLATE_FN)
INIT_FUNCTION_LIST_FOR_OP_TEMPLATE(single_add_op_template_fns, SINGLE_ADD_FN_NAME)

/* SUB FUNCTIONS IMPLEMENTATION */
#define SINGLE_SUB_FN_NAME(_ftype, _stype) single_sub_op_template_fn_ ## _ftype ## _ ## _stype
#define SINGLE_SUB_OP_TEMPLATE_FN(_ftype, _stype)  \
static void SINGLE_SUB_FN_NAME(_ftype, _stype)(const void *first, const void *second, void *output){ \
    ((_ftype *)output)[0] = (_ftype)(((_ftype *)first)[0] - ((_stype *)second)[0]); \
}
IMPLEMENT_FUNCTIONS_FOR_OP_TEMPLATE(SINGLE_SUB_OP_TEMPLATE_FN)
INIT_FUNCTION_LIST_FOR_OP_TEMPLATE(single_sub_op_template_fns, SINGLE_SUB_FN_NAME)

/* RELU FUNCTIONS IMPLEMENTATION */
#define SINGLE_RELU_FN_NAME(_ftype, _stype) single_relu_op_template_fn_ ## _ftype ## _ ## _stype
#define SINGLE_RELU_OP_TEMPLATE_FN(_ftype, _stype)  \
static void SINGLE_RELU_FN_NAME(_ftype, _stype)(const void *first, const void *second, void *output){ \
    (void)second; \
    ((_ftype *)output)[0] = (((_ftype *)first)[0] > (_ftype)0) ? ((_ftype *)first)[0] : (_ftype)0; \
}
IMPLEMENT_FUNCTIONS_FOR_OP_TEMPLATE(SINGLE_RELU_OP_TEMPLATE_FN)
INIT_FUNCTION_LIST_FOR_OP_TEMPLATE(single_relu_op_template_fns, SINGLE_RELU_FN_NAME)

/* CONV FUNCTIONS IMPLEMENTATION */
#define SINGLE_CONV_FN_NAME(_ftype, _stype) single_conv_op_template_fn_ ## _ftype ## _ ## _stype
#define SINGLE_CONV_OP_TEMPLATE_FN(_ftype, _stype)  \
static void SINGLE_CONV_FN_NAME(_ftype, _stype)(const void *first, const void *second, void *output){ \
    ((_ftype *)output)[0] += (_ftype)(((_ftype *)first)[0] * ((_stype *)second)[0]); \
}
IMPLEMENT_FUNCTIONS_FOR_OP_TEMPLATE(SINGLE_CONV_OP_TEMPLATE_FN)
INIT_FUNCTION_LIST_FOR_OP_TEMPLATE(single_conv_op_template_fns, SINGLE_CONV_FN_NAME)

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

static void adci_check_tensor_broadcast(const struct adci_tensor *tensor, const struct adci_tensor *second){
    const unsigned length = (tensor->n_dimension < second->n_dimension) ? tensor->n_dimension : second->n_dimension;
    for(unsigned int i = 0; i < length; i++){
        ADCI_ASSERT(
            tensor->shape[tensor->n_dimension - i - 1] == 1 ||
            second->shape[second->n_dimension - i - 1] == 1 ||
            tensor->shape[tensor->n_dimension - i - 1] == second->shape[second->n_dimension - i - 1]
        );
    }
}

static void adci_generic_broadcast_op(struct adci_vector inputs, struct adci_tensor *output, single_op_template_fn_t op){
    ADCI_ASSERT(inputs.length == 2);
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    struct adci_tensor *operand = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    adci_check_tensor_broadcast(tensor, operand);
    /* BROADCASTING VALID */
    struct adci_tensor temp = *tensor;
    if(tensor == output) output->data = NULL;
    adci_prepare_output_tensor(tensor->shape, tensor->n_dimension, tensor->dtype, output);
    const unsigned int volume = adci_tensor_element_count(&temp);
    const unsigned int operand_volume = adci_tensor_element_count(operand);
    const unsigned int element_size = adci_tensor_dtype_size(temp.dtype);
    const unsigned int operand_element_size = adci_tensor_dtype_size(operand->dtype);
    for(unsigned int i = 0; i < volume; i++)
        op(temp.data + i * element_size, operand->data + (i % operand_volume) * operand_element_size, output->data + i * element_size);
    if(tensor == output) ADCI_FREE(temp.data);
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

static void adci_single_channel_conv(
    struct adci_tensor *tensor, 
    struct adci_tensor *filter, 
    struct adci_tensor *stride, 
    struct adci_tensor *dims,
    struct adci_tensor *output,
    unsigned int input_batch_index,
    unsigned int filter_channel_index)
    {
    const unsigned int filter_dims[] = {1, 2, 3};
    const unsigned int element_size = adci_tensor_dtype_size(output->dtype);
    const unsigned int output_batch_volume = adci_tensor_element_count_ext(output->n_dimension - 1, output->shape + 1);
    const unsigned int tensor_batch_volume = adci_tensor_element_count_ext(tensor->n_dimension - 1, tensor->shape + 1);
    const unsigned int num_convolutions = output->shape[adci_tensor_get_i32(dims, 0)] * output->shape[adci_tensor_get_i32(dims, 1)]; 
    const unsigned int filter_volume = adci_tensor_element_count_ext(filter->n_dimension - 1, filter->shape + 1);
    struct adci_multi_dim_counter output_counter = adci_tensor_init_multidim_counter(output, dims->data, dims->shape[0]);
    struct adci_multi_dim_counter tensor_counter = adci_tensor_init_multidim_counter(tensor, dims->data, dims->shape[0]);
    single_op_template_fn_t conv_op = single_conv_op_template_fns[tensor->dtype * ADCI_NUM_SUPPORTED_TYPES + filter->dtype];
    for(unsigned int i = 0; i < num_convolutions; i++){
        const unsigned int curr_inner_offset = adci_tensor_get_counter_offset(output_counter);
        const unsigned int current_offset = (curr_inner_offset + input_batch_index * output_batch_volume) * element_size;
        adci_reset_value(output->data + current_offset, output->dtype);
        const unsigned int curr_tensor_offset = input_batch_index * tensor_batch_volume;
        struct adci_multi_dim_counter filter_counter = adci_tensor_init_multidim_counter(filter, filter_dims, sizeof(filter_dims) / sizeof(unsigned int));
        for(unsigned int j = 0; j < filter_volume; j++){
            /* COMPUTE OFFSET FOR FILTER COORDS ON INPUT TENSOR */
            const unsigned int filter_offset = (filter_channel_index * filter_volume + adci_tensor_get_counter_offset(filter_counter)) * adci_tensor_dtype_size(filter->dtype);
            unsigned int tensor_offset = curr_tensor_offset;
            tensor_offset += output_counter.counter[1] * adci_tensor_get_i32(stride, 1) * tensor_counter.precomputed_volumes[tensor_counter.dim_indeces[1]];
            tensor_offset += output_counter.counter[0] * adci_tensor_get_i32(stride, 0) * tensor_counter.precomputed_volumes[tensor_counter.dim_indeces[0]];
            for(unsigned int k = 0; k < filter_counter.free_dims_count; k++) tensor_offset += filter_counter.counter[k] * tensor_counter.precomputed_volumes[filter_counter.dim_indeces[k]];
            conv_op(tensor->data + tensor_offset * element_size, filter->data + filter_offset, output->data + current_offset);
            adci_tensor_increase_counter(&filter_counter);
        }
        adci_tensor_increase_counter(&output_counter);
    }
}

static void adci_compute_pool2d(
    const struct adci_tensor *tensor, 
    const struct adci_tensor *size,
    const struct adci_tensor *stride,
    const struct adci_tensor *dims,
    struct adci_tensor *output,
    const struct adci_multi_dim_counter free_tensor_counter,
    const struct adci_multi_dim_counter free_output_counter,
    unsigned int current_out_width, unsigned int current_out_height,
    single_op_template_fn_t pool_op)
    {
    const unsigned int width = adci_tensor_get_i32(size, 0);
    const unsigned int height = adci_tensor_get_i32(size, 1);
    unsigned int tensor_offset = adci_tensor_get_counter_offset(free_tensor_counter);
    const unsigned int current_width = current_out_width * adci_tensor_get_i32(stride, 0);
    const unsigned int current_height = current_out_height * adci_tensor_get_i32(stride, 1);
    tensor_offset += current_width * free_tensor_counter.precomputed_volumes[adci_tensor_get_i32(dims, 0)];
    tensor_offset += current_height * free_tensor_counter.precomputed_volumes[adci_tensor_get_i32(dims, 1)];
    unsigned int output_offset = adci_tensor_get_counter_offset(free_output_counter);
    output_offset += current_out_width * free_output_counter.precomputed_volumes[adci_tensor_get_i32(dims, 0)];
    output_offset += current_out_height * free_output_counter.precomputed_volumes[adci_tensor_get_i32(dims, 1)];
    const unsigned int element_size = adci_tensor_dtype_size(tensor->dtype);
    void *output_data = output->data + output_offset * element_size;
    adci_reset_value(output_data, output->dtype);
    const unsigned int width_volume = free_tensor_counter.precomputed_volumes[adci_tensor_get_i32(dims, 0)];
    const unsigned int height_volume = free_tensor_counter.precomputed_volumes[adci_tensor_get_i32(dims, 1)];
    for(unsigned int i = 0; i < width; i++){
        const unsigned int curr_inner_tensor_width_offset = i * width_volume;
        for(unsigned int j = 0; j < height; j++){
            const unsigned int curr_inner_tensor_height_offset = j * height_volume;
            pool_op(tensor->data + (tensor_offset + curr_inner_tensor_width_offset + curr_inner_tensor_height_offset) * element_size, output_data, output_data);
        }
    }
}

/* END PRIVATE FUNCTIONS */

void ADCI_EXIT_POINT adci_tensor_add(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 2);
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    struct adci_tensor *operand = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    single_op_template_fn_t add_op = single_add_op_template_fns[tensor->dtype * ADCI_NUM_SUPPORTED_TYPES + operand->dtype];
    adci_generic_broadcast_op(inputs, output, add_op);
}

void ADCI_EXIT_POINT adci_tensor_sub(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 2);
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    struct adci_tensor *operand = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    single_op_template_fn_t sub_op = single_sub_op_template_fns[tensor->dtype * ADCI_NUM_SUPPORTED_TYPES + operand->dtype];
    adci_generic_broadcast_op(inputs, output, sub_op);
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
    const struct adci_tensor *padding = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
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
    adci_check_tensor_broadcast(element, parameters);
    /* TENSORS CAN BE BROADCAST */
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

void ADCI_EXIT_POINT adci_tensor_relu(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 1);
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    const unsigned int volume = adci_tensor_element_count(tensor);
    const unsigned int element_size = adci_tensor_dtype_size(tensor->dtype);
    struct adci_tensor temp = *tensor;
    if(tensor == output) output->data = NULL;
    adci_prepare_output_tensor(tensor->shape, tensor->n_dimension, tensor->dtype, output);
    single_op_template_fn_t relu_op = single_relu_op_template_fns[tensor->dtype * ADCI_NUM_SUPPORTED_TYPES];
    for(unsigned int i = 0; i < volume; i++){
        const unsigned int offset = i * element_size; 
        relu_op(temp.data + offset, NULL, output->data + offset);
    }
    if(tensor == output) ADCI_FREE(temp.data);
}

void ADCI_EXIT_POINT adci_tensor_cast(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 2);
    struct adci_tensor *input = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    struct adci_tensor *cast_dtype = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    ADCI_ASSERT(cast_dtype->dtype == ADCI_I32 && cast_dtype->n_dimension == 1 && cast_dtype->shape[0] == 1);
    enum adci_tensor_type output_type = (enum adci_tensor_type)adci_tensor_get_i32(cast_dtype, 0);
    const unsigned int in_elem_size = adci_tensor_dtype_size(input->dtype);
    const unsigned int ou_elem_size = adci_tensor_dtype_size(output_type);
    struct adci_tensor temp = *input;
    if(input == output) output->data = NULL;
    const unsigned int output_size = adci_prepare_output_tensor(input->shape, input->n_dimension, output_type, output);
    unsigned int volume = output_size / ou_elem_size;
    const single_op_template_fn_t cast = single_cast_op_template_fns[input->dtype * ADCI_NUM_SUPPORTED_TYPES + output->dtype]; 
    for(unsigned int i = 0; i < volume; i++)
        cast(input->data + i * in_elem_size, NULL, output->data + i * ou_elem_size);
    if(input == output) ADCI_FREE(temp.data);
}

void ADCI_EXIT_POINT adci_tensor_softmax(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 2);
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
    int8_t maximum[element_size];
    struct adci_multi_dim_counter free_dim_counter = adci_tensor_init_multidim_counter(&temp, (unsigned int *)free_dim_mapping.data, free_dim_mapping.length);
    for(unsigned int i = 0; i < out_volume; i++){
        unsigned int fixed_offset = adci_tensor_get_counter_offset(free_dim_counter);
        adci_reset_value(maximum, tensor->dtype);
        struct adci_multi_dim_counter reduced_dim_counter = adci_tensor_init_multidim_counter(&temp, (unsigned int *)reduced_dim_mapping.data, reduced_dim_mapping.length);
        for(unsigned int j = 0; j < reduce_volume; j++){
            /* CHECK FOR MAXIMUM VALUE AMONGST THE REDUCED DIMENSIONS */
            unsigned int reduced_offset = adci_tensor_get_counter_offset(reduced_dim_counter);
            int8_t *current = (int8_t *)temp.data + (fixed_offset + reduced_offset) * element_size;
            adci_compare_max(current, maximum, temp.dtype);
            adci_tensor_increase_counter(&reduced_dim_counter);
        }
        memcpy((int8_t *)output->data + i * element_size, maximum, element_size);
        adci_tensor_increase_counter(&free_dim_counter);
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
    adci_check_tensor_broadcast(tensor, mult);
    /* BROADCASTING VALID */
    struct adci_tensor temp = *tensor;
    if(tensor == output) output->data = NULL;
    adci_prepare_output_tensor(tensor->shape, tensor->n_dimension, tensor->dtype, output);
    const unsigned int volume = adci_tensor_element_count(&temp);
    const unsigned int mult_volume = adci_tensor_element_count(mult);
    const unsigned int element_size = adci_tensor_dtype_size(temp.dtype);
    const unsigned int mult_element_size = adci_tensor_dtype_size(mult->dtype);
    single_op_template_fn_t single_mult = single_mult_op_template_fns[temp.dtype * ADCI_NUM_SUPPORTED_TYPES + mult->dtype];
    for(unsigned int i = 0; i < volume; i++)
        single_mult(temp.data + i * element_size, mult->data + (i % mult_volume) * mult_element_size, output->data + i * element_size);
    if(tensor == output) ADCI_FREE(temp.data);
}

void ADCI_EXIT_POINT adci_tensor_max_pool2D(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 4);
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0); 
    /* WIDTH, HEIGHT ORDER */
    struct adci_tensor *size   = *(struct adci_tensor **)adci_vector_get(&inputs, 1); 
    struct adci_tensor *stride = *(struct adci_tensor **)adci_vector_get(&inputs, 2);
    /* SPECIFIES THE 2 DIMENSIONS TO BE CONSIDERED FOR THE 2D POOL [WIDTH_DIM, HEIGHT_DIM]*/
    struct adci_tensor *dims   = *(struct adci_tensor **)adci_vector_get(&inputs, 3);
    ADCI_ASSERT(dims->n_dimension == 1 && dims->shape[0] == 2);
    ADCI_ASSERT(size->n_dimension == 1 && size->shape[0] == 2);
    ADCI_ASSERT(stride->n_dimension == 1 && stride->shape[0] == 2);
    ADCI_ASSERT(size->dtype == ADCI_I32 && stride->dtype == ADCI_I32 && dims->dtype == ADCI_I32);
    /* COMPUTE OUTPUT SHAPE */
    unsigned int shape[tensor->n_dimension];
    memcpy(shape, tensor->shape, sizeof(shape));
    for(unsigned int i = 0; i < dims->shape[0]; i++){
        unsigned int current = adci_tensor_get_i32(dims, i); 
        shape[current] = ((tensor->shape[current] - adci_tensor_get_i32(size, i)) / adci_tensor_get_i32(stride, i)) + 1;
    }
    struct adci_tensor temp = *tensor;
    if(tensor == output) output->data = NULL;
    unsigned int output_volume = adci_prepare_output_tensor(shape, tensor->n_dimension, tensor->dtype, output) / adci_tensor_dtype_size(tensor->dtype);
    /* WE WANT A COUNTER ARROUND THE FREE DIMENSIONS (DIMENSIONS NOT CONCERNED BY THE 2D POOL) */
    unsigned int index = 0;
    unsigned int free_dims[tensor->n_dimension - 2];
    for(int i = 0; i < (int)tensor->n_dimension; i++)
        if(adci_tensor_get_i32(dims, 0) != i && adci_tensor_get_i32(dims, 1) != i) free_dims[index++] = i;
    struct adci_multi_dim_counter output_counter = adci_tensor_init_multidim_counter(output, free_dims, dims->shape[0]);
    struct adci_multi_dim_counter tensor_counter = adci_tensor_init_multidim_counter(tensor, free_dims, dims->shape[0]);
    const unsigned int matrix_count = output_volume / (output->shape[adci_tensor_get_i32(dims, 0)] * output->shape[adci_tensor_get_i32(dims, 1)]);
    single_op_template_fn_t max_op = single_max_op_template_fns[tensor->dtype * ADCI_NUM_SUPPORTED_TYPES + tensor->dtype];
    for(unsigned int i = 0; i < matrix_count; i++){
        for(unsigned int width = 0; width < output->shape[adci_tensor_get_i32(dims, 0)]; width++){
            for(unsigned int height = 0; height < output->shape[adci_tensor_get_i32(dims, 1)]; height++)
                adci_compute_pool2d(tensor, size, stride, dims, output, tensor_counter, output_counter, width, height, max_op);
        }
        /* PASS TO NEXT 2D PLANE */
        adci_tensor_increase_counter(&output_counter);
        adci_tensor_increase_counter(&tensor_counter);
    }
    if(tensor == output) ADCI_FREE(temp.data);
}

/*@stride: WIDTH, HEIGHT ORDER */
/*@dims: SPECIFIES THE INDECES OF DIMENSIONS TO BE CONSIDERED FOR THE 2D MASK [WIDTH_DIM, HEIGHT_DIM, CHANNEL_DIM] */

void ADCI_EXIT_POINT adci_tensor_conv2D_args(
    struct adci_tensor *tensor,
    struct adci_tensor *filter,
    struct adci_tensor *stride,
    struct adci_tensor *dims,
    struct adci_tensor *output)
    {
    ADCI_ASSERT(tensor->n_dimension == filter->n_dimension);
    ADCI_ASSERT(stride->n_dimension == 1 && stride->shape[0] == 2);
    ADCI_ASSERT(dims->n_dimension == 1 && dims->shape[0] == 3);
    ADCI_ASSERT(stride->dtype == ADCI_I32 && dims->dtype == ADCI_I32);
    /* COMPUTE OUTPUT SHAPE */
    unsigned int shape[tensor->n_dimension];
    memcpy(shape, tensor->shape, sizeof(shape));
    const unsigned int channel_index = adci_tensor_get_i32(dims, 2);
    for(unsigned int i = 0; i < dims->shape[0] - 1; i++){
        const unsigned int current_index = adci_tensor_get_i32(dims, i);
        shape[current_index] = ((tensor->shape[current_index] - filter->shape[current_index]) / adci_tensor_get_i32(stride, i)) + 1;
    }
    shape[channel_index] = filter->shape[0];
    struct adci_tensor temp = *tensor;
    if(tensor == output) output->data = NULL;
    adci_prepare_output_tensor(shape, temp.n_dimension, temp.dtype, output);
    for(unsigned int i = 0; i < output->shape[0]; i++){
        for(unsigned int channel = 0; channel < filter->shape[0]; channel++){
            /* RUN CONV ON CURRENT 2D/3D VOLUME */
            adci_single_channel_conv(&temp, filter, stride, dims, output, i, channel);
        }
    }
    if(tensor == output) ADCI_FREE(temp.data);
}


/* FILTER TENSOR HAS TO BE IN THE SHAPE [OUT_CHANNEL, WIDTH, HEIGHT, IN_CHANNEL]*/
void ADCI_EXIT_POINT adci_tensor_conv2D(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 4);
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0); 
    struct adci_tensor *filter = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    struct adci_tensor *stride = *(struct adci_tensor **)adci_vector_get(&inputs, 2);
    struct adci_tensor *dims = *(struct adci_tensor **)adci_vector_get(&inputs, 3);
    adci_tensor_conv2D_args(tensor, filter, stride, dims, output);
}

void ADCI_EXIT_POINT adci_tensor_transpose_args(struct adci_tensor *tensor, struct adci_tensor *dims, struct adci_tensor *output){
    ADCI_ASSERT(dims->n_dimension == 1 && dims->shape[0] == tensor->n_dimension && dims->dtype == ADCI_I32);
    unsigned int shape[tensor->n_dimension];
    for(unsigned int i = 0; i < tensor->n_dimension; i++)
        shape[i] = tensor->shape[adci_tensor_get_i32(dims, i)];
    struct adci_tensor temp = *tensor;
    if(tensor == output) output->data = NULL;
    const unsigned int element_size = adci_tensor_dtype_size(tensor->dtype);
    const unsigned int volume = adci_prepare_output_tensor(shape, tensor->n_dimension, tensor->dtype, output) / element_size;
    struct adci_multi_dim_counter input_counter = adci_tensor_init_multidim_counter(&temp, NULL, temp.n_dimension);
    /* JUST TO GET THE PRECOMPUTED VOLUMES */
    struct adci_multi_dim_counter output_counter = adci_tensor_init_multidim_counter(output, NULL, output->n_dimension);
    unsigned int *counter[temp.n_dimension];
    for(unsigned int i = 0; i < tensor->n_dimension; i++)
        counter[i] = &input_counter.counter[adci_tensor_get_i32(dims, i)];
    for(unsigned int i = 0; i < volume; i++){
        const unsigned int input_offset = i * element_size;
        /* GET OUTPUT OFFSET*/
        unsigned int output_offset = 0;
        for(unsigned int j = 0; j < temp.n_dimension; j++)
            output_offset += *counter[j] * output_counter.precomputed_volumes[j];
        memcpy(output->data + output_offset * element_size, temp.data + input_offset, element_size);
        adci_tensor_increase_counter(&input_counter);
    }
    if(tensor == output) ADCI_FREE(temp.data);
}

void ADCI_EXIT_POINT adci_tensor_transpose(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 2);
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0); 
    struct adci_tensor *dims = *(struct adci_tensor **)adci_vector_get(&inputs, 1); 
    adci_tensor_transpose_args(tensor, dims, output);
}

void ADCI_EXIT_POINT adci_tensor_fully_connected(struct adci_vector inputs, struct adci_tensor *output){
    struct adci_tensor *input = *(struct adci_tensor **)adci_vector_get(&inputs, 0); 
    struct adci_tensor *weights = *(struct adci_tensor **)adci_vector_get(&inputs, 1); 
    ADCI_ASSERT(input->n_dimension == 2 || input->n_dimension == 1);
    ADCI_ASSERT(weights->n_dimension == 2);
    /* MAKE SURE VECTOR/MATRIX MULTIPLICATION IS VALID */
    ADCI_ASSERT(input->shape[input->n_dimension - 1] == weights->shape[1]);
    /* COMPUTE OUTPUT SIZE */
    unsigned int shape[2];
    shape[0] = input->shape[0];
    shape[1] = weights->shape[0];
    struct adci_tensor temp = *input;
    if(input == output) output->data = NULL;
    adci_prepare_output_tensor(shape, sizeof(shape) / sizeof(unsigned int), temp.dtype, output);
    const unsigned int element_size = adci_tensor_dtype_size(temp.dtype);
    single_op_template_fn_t multiplier = single_mult_op_template_fns[temp.dtype * ADCI_NUM_SUPPORTED_TYPES + weights->dtype];
    /* THE ELEMENTS TO BE ADDED ARE THE OUTPUT OF THE MULTIPLY FUNCTION WHICH WILL BE CASTED TO THE TYPE OF THE INPUT */
    single_op_template_fn_t adder = single_add_op_template_fns[temp.dtype * ADCI_NUM_SUPPORTED_TYPES + temp.dtype];
    for(unsigned int batch = 0; batch < temp.shape[0]; batch++){
        for(unsigned int row = 0; row < weights->shape[0]; row++){
            /* COMPUTE DOT PRODUCT FOR CURRENT WEIGHT LINE */
            const unsigned int output_offset = row * element_size;
            adci_reset_value(output->data + output_offset, temp.dtype);
            int8_t temp_mult[element_size];
            const unsigned int weights_outer_offset = row * weights->shape[1];
            for(unsigned int col = 0; col < weights->shape[1]; col++){
                const unsigned int vector_offset = col * element_size;
                const unsigned int weights_offset = (weights_outer_offset + col) * element_size;
                multiplier(temp.data + vector_offset, weights->data + weights_offset, temp_mult);
                adder(output->data + output_offset, temp_mult, output->data + output_offset);
            }
        }
    }
    if(input == output) ADCI_FREE(temp.data);
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
    case ADCI_TENSOR_RELU: return adci_tensor_relu(inputs, output);
    case ADCI_TENSOR_CAST: return adci_tensor_cast(inputs, output);
    case ADCI_TENSOR_SOFTMAX: return adci_tensor_softmax(inputs, output);
    case ADCI_TENSOR_REDUCE_MAX: return adci_tensor_reduce_max(inputs, output);
    case ADCI_TENSOR_CONCAT: return adci_tensor_concat(inputs, output);
    case ADCI_TENSOR_MUL: return adci_tensor_mul(inputs, output);
    case ADCI_TENSOR_MAX_POOL2D:  return adci_tensor_max_pool2D(inputs, output);
    case ADCI_TENSOR_CONV2D: return adci_tensor_conv2D(inputs, output);
    case ADCI_TENSOR_TRANSPOSE: return adci_tensor_transpose(inputs, output);
    case ADCI_TENSOR_FULLY_CONNECTED: return adci_tensor_fully_connected(inputs, output);
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
    