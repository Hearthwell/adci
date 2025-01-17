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
    unsigned int batch_index,
    unsigned int batch_field_index,
    unsigned int filter_channel_index)
    {
    const unsigned int element_size = adci_tensor_dtype_size(output->dtype);
    const unsigned int output_batch_volume = adci_tensor_element_count_ext(output->n_dimension - batch_field_index - 1, output->shape + batch_field_index + 1);
    const unsigned int tensor_batch_volume = adci_tensor_element_count_ext(tensor->n_dimension - batch_field_index - 1, tensor->shape + batch_field_index + 1);
    const unsigned int num_convolutions = output->shape[adci_tensor_get_i32(dims, 0)] * output->shape[adci_tensor_get_i32(dims, 1)]; 
    const unsigned int filter_volume = adci_tensor_element_count_ext(filter->n_dimension - 1, filter->shape + 1);
    const unsigned int output_outer_offset = batch_index * output_batch_volume;
    /* WE ONLY WANT A COUNTER FOR THE WIDTH AND HEIGHT (THE NUMBER OF CONVOLUTIONS TO PERFORM) */
    struct adci_multi_dim_counter output_counter = adci_tensor_init_multidim_counter(output, dims->data, dims->shape[0] - 1);
    /* ONLY USED FOR PRECOMPUTED VOLUMES */
    struct adci_multi_dim_counter tensor_counter = adci_tensor_init_multidim_counter(tensor, dims->data, dims->shape[0]);
    single_op_template_fn_t conv_op = single_conv_op_template_fns[tensor->dtype * ADCI_NUM_SUPPORTED_TYPES + filter->dtype];
    for(unsigned int i = 0; i < num_convolutions; i++){
        const unsigned int output_inner_offset = adci_tensor_get_counter_offset(output_counter) + filter_channel_index * output_counter.precomputed_volumes[adci_tensor_get_i32(dims, 2)];
        const unsigned int current_offset = (output_inner_offset + output_outer_offset) * element_size;
        adci_reset_value(output->data + current_offset, output->dtype);
        const unsigned int curr_tensor_offset = batch_index * tensor_batch_volume;
        /* FILTER IS IN FORMAT OUT_CHANNEL, WIDTH, HEIGHT, IN_CHANNEL */
        struct adci_multi_dim_counter filter_counter = adci_tensor_alldim_counter_except(filter, 0);
        for(unsigned int j = 0; j < filter_volume; j++){
            /* COMPUTE OFFSET FOR FILTER COORDS ON INPUT TENSOR */
            const unsigned int filter_offset = (filter_channel_index * filter_volume + adci_tensor_get_counter_offset(filter_counter)) * adci_tensor_dtype_size(filter->dtype);
            unsigned int tensor_offset = curr_tensor_offset;
            tensor_offset += output_counter.counter[1] * adci_tensor_get_i32(stride, 1) * tensor_counter.precomputed_volumes[tensor_counter.dim_indeces[1]];
            tensor_offset += output_counter.counter[0] * adci_tensor_get_i32(stride, 0) * tensor_counter.precomputed_volumes[tensor_counter.dim_indeces[0]];
            for(unsigned int k = 0; k < filter_counter.free_dims_count; k++) tensor_offset += filter_counter.counter[k] * tensor_counter.precomputed_volumes[tensor_counter.dim_indeces[k]];
            conv_op(tensor->data + tensor_offset * element_size, filter->data + filter_offset, output->data + current_offset);
            adci_tensor_increase_counter(&filter_counter);
        }
        adci_tensor_increase_counter(&output_counter);
    }
}

/* RETURNS WHERE THE POOL RESUTL HAS BEEN WRITTEN IN OUTPUT TENSOR */
static void * adci_compute_pool2d(
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
    /* RESET OUTPUT TO FIRST ELEM */
    memcpy(output_data, tensor->data + tensor_offset * element_size, element_size);
    const unsigned int width_volume = free_tensor_counter.precomputed_volumes[adci_tensor_get_i32(dims, 0)];
    const unsigned int height_volume = free_tensor_counter.precomputed_volumes[adci_tensor_get_i32(dims, 1)];
    for(unsigned int i = 0; i < width; i++){
        const unsigned int curr_inner_tensor_width_offset = i * width_volume;
        for(unsigned int j = 0; j < height; j++){
            /* SKIP FIRST ELEMENT SINCE IT IS THE RESET VALUE */
            if(i == 0 && j == 0) continue;
            const unsigned int curr_inner_tensor_height_offset = j * height_volume;
            pool_op(tensor->data + (tensor_offset + curr_inner_tensor_width_offset + curr_inner_tensor_height_offset) * element_size, output_data, output_data);
        }
    }
    return output_data;
}

/* POOL FUNCTIONS IMPLEMENTATION */
typedef void (*adci_pool2D_op)(
    const struct adci_tensor *tensor, 
    const struct adci_tensor *size, 
    const struct adci_tensor *stride, 
    const struct adci_tensor *dims,
    struct adci_tensor *output,
    struct adci_multi_dim_counter tensor_counter,
    struct adci_multi_dim_counter output_counter,
    unsigned int width, unsigned int height);

static void adci_tensor_max_pool2D_op(
    const struct adci_tensor *tensor, 
    const struct adci_tensor *size, 
    const struct adci_tensor *stride, 
    const struct adci_tensor *dims,
    struct adci_tensor *output,
    struct adci_multi_dim_counter tensor_counter,
    struct adci_multi_dim_counter output_counter,
    unsigned int width, unsigned int height
){
    single_op_template_fn_t max_op = single_max_op_template_fns[tensor->dtype * ADCI_NUM_SUPPORTED_TYPES + tensor->dtype];
    (void)adci_compute_pool2d(tensor, size, stride, dims, output, tensor_counter, output_counter, width, height, max_op);
}

static void adci_tensor_avg_pool2D_op(
    const struct adci_tensor *tensor, 
    const struct adci_tensor *size, 
    const struct adci_tensor *stride, 
    const struct adci_tensor *dims,
    struct adci_tensor *output,
    struct adci_multi_dim_counter tensor_counter,
    struct adci_multi_dim_counter output_counter,
    unsigned int width, unsigned int height
){
    single_op_template_fn_t add_op = single_add_op_template_fns[tensor->dtype * ADCI_NUM_SUPPORTED_TYPES + tensor->dtype];
    void *sum = adci_compute_pool2d(tensor, size, stride, dims, output, tensor_counter, output_counter, width, height, add_op);
    const float factor = 1.0f / (adci_tensor_get_i32(size, 0) * adci_tensor_get_i32(size, 1));
    single_op_template_fn_t mult_op = single_mult_op_template_fns[tensor->dtype * ADCI_NUM_SUPPORTED_TYPES + ADCI_F32];
    mult_op(sum, &factor, sum);
}

/* GET OUTPUT FORMAT FOR NON-TRIVIAL OPERATIONS */
struct adci_output_format{
    unsigned int n_dimension;
    unsigned int shape[ADCI_TENSOR_MAX_DIM];
    void *data;
};

static struct adci_output_format adci_tensor_op_padding_format(struct adci_vector inputs){
    struct adci_tensor *element = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    const struct adci_tensor *padding = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    ADCI_ASSERT(padding->dtype == ADCI_I32);
    ADCI_ASSERT(padding->n_dimension == 2);
    ADCI_ASSERT(element->n_dimension == padding->shape[0]);
    unsigned int output_shape[element->n_dimension];
    for(unsigned int i = 0; i < padding->shape[0]; i++){
        const unsigned int pad_size = ((int32_t *)padding->data)[i * padding->shape[1]] + ((int32_t *)padding->data)[i * padding->shape[1] + 1];
        output_shape[i] = element->shape[i] + pad_size;
    }
    struct adci_output_format format = {.n_dimension = element->n_dimension, .data = NULL};
    memcpy(format.shape, output_shape, element->n_dimension * sizeof(uint32_t));
    return format;
}

static struct adci_output_format adci_tensor_op_reshape_format(struct adci_vector inputs){
    ADCI_ASSERT(inputs.length == 2);
    struct adci_tensor *shape = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    ADCI_ASSERT(shape->n_dimension == 1);
    ADCI_ASSERT(shape->data != NULL);
    ADCI_ASSERT(shape->dtype == ADCI_I32);
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    (void) tensor;
    ADCI_ASSERT(adci_tensor_element_count(tensor) == adci_tensor_element_count_ext(shape->shape[0], shape->data));
    struct adci_output_format output_format = {.n_dimension = shape->shape[0]};
    memcpy(output_format.shape, shape->data, output_format.n_dimension * sizeof(uint32_t));
    return output_format;
}

static struct adci_output_format adci_tensor_op_pool2D_format(struct adci_tensor *tensor, struct adci_tensor *size, struct adci_tensor *stride, struct adci_tensor *dims){
    ADCI_ASSERT(dims->n_dimension == 1 && dims->shape[0] == 2);
    ADCI_ASSERT(size->n_dimension == 1 && size->shape[0] == 2);
    ADCI_ASSERT(stride->n_dimension == 1 && stride->shape[0] == 2);
    ADCI_ASSERT(size->dtype == ADCI_I32 && stride->dtype == ADCI_I32 && dims->dtype == ADCI_I32);
    /* COMPUTE OUTPUT SHAPE */
    struct adci_output_format output_format = {.n_dimension = tensor->n_dimension};
    memcpy(output_format.shape, tensor->shape, tensor->n_dimension * sizeof(uint32_t));
    for(unsigned int i = 0; i < dims->shape[0]; i++){
        unsigned int current = adci_tensor_get_i32(dims, i); 
        output_format.shape[current] = ((tensor->shape[current] - adci_tensor_get_i32(size, i)) / adci_tensor_get_i32(stride, i)) + 1;
    }
    return output_format;
}

static struct adci_output_format adci_tensor_op_concat_format(struct adci_vector inputs){
    const struct adci_tensor *axis = *(struct adci_tensor **)adci_vector_get(&inputs, inputs.length - 1); 
    ADCI_ASSERT(axis->n_dimension == 1);
    ADCI_ASSERT(axis->shape[0] == 1);
    ADCI_ASSERT(axis->dtype == ADCI_I32);
    const unsigned int dim = adci_tensor_get_i32(axis, 0);
    struct adci_tensor *first = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    ADCI_ASSERT(dim < first->n_dimension);
    unsigned int concat_dim_size = first->shape[dim];
    for(unsigned int i = 1; i < inputs.length - 1; i++){
        /* MAKE SURE ALL TENSORS TO BE CONCATENATED HAVE SAME SHAPE EXCEPT FOR THE DIM TO BE CONCATENATED */
        struct adci_tensor *current = *(struct adci_tensor **)adci_vector_get(&inputs, i);
        ADCI_ASSERT(first->n_dimension == current->n_dimension);
        ADCI_ASSERT(first->dtype == current->dtype);
        for(unsigned int j = 0; j < current->n_dimension; j++){
            if(j == dim){
                concat_dim_size += current->shape[j];
                continue;
            } 
            ADCI_ASSERT(current->shape[j] == first->shape[j]);
        }
    }
    /* ALL THE DIMS MATCH EXCEPT FOR THE CONCATENATED DIM */
    struct adci_output_format output_format = {.n_dimension = first->n_dimension};
    memcpy(output_format.shape, first->shape, first->n_dimension * sizeof(uint32_t));
    output_format.shape[dim] = concat_dim_size;
    return output_format;
}

static struct adci_output_format adci_tensor_op_conv2D_format(    
    struct adci_tensor *tensor,
    struct adci_tensor *filter,
    struct adci_tensor *stride,
    struct adci_tensor *dims)
    {
    ADCI_ASSERT(tensor->n_dimension == filter->n_dimension);
    ADCI_ASSERT(stride->n_dimension == 1 && stride->shape[0] == 2);
    ADCI_ASSERT(dims->n_dimension == 1 && dims->shape[0] == 3);
    ADCI_ASSERT(stride->dtype == ADCI_I32 && dims->dtype == ADCI_I32);
    /* COMPUTE OUTPUT SHAPE */
    struct adci_output_format output_format = {.n_dimension = tensor->n_dimension};
    memset(output_format.shape, 0, sizeof(output_format.shape));
    const unsigned int channel_index = adci_tensor_get_i32(dims, 2);
    for(unsigned int i = 0; i < dims->shape[0] - 1; i++){
        const unsigned int current_index = adci_tensor_get_i32(dims, i);
        output_format.shape[current_index] = ((tensor->shape[current_index] - filter->shape[current_index]) / adci_tensor_get_i32(stride, i)) + 1;
    }
    output_format.shape[channel_index] = filter->shape[0];
    /* GET BATCH DIM */
    unsigned int batch_dim = 0;
    for(;batch_dim < tensor->n_dimension; batch_dim++)
        if(output_format.shape[batch_dim] == 0) break;
    output_format.shape[batch_dim] = tensor->shape[batch_dim];
    output_format.data = ADCI_ALLOC(sizeof(unsigned int));
    *(unsigned int *)output_format.data = batch_dim;
    return output_format;
}

static struct adci_output_format adci_tensor_op_transpose_format(struct adci_tensor *tensor, struct adci_tensor *dims){
    ADCI_ASSERT(dims->n_dimension == 1 && dims->shape[0] == tensor->n_dimension && dims->dtype == ADCI_I32);
    struct adci_output_format output_format = {.n_dimension = tensor->n_dimension};
    for(unsigned int i = 0; i < tensor->n_dimension; i++)
        output_format.shape[i] = tensor->shape[adci_tensor_get_i32(dims, i)];
    return output_format;
}

static struct adci_output_format adci_tensor_op_fully_connected_format(struct adci_tensor *input, struct adci_tensor *weights){
    ADCI_ASSERT(input->n_dimension == 2 || input->n_dimension == 1);
    ADCI_ASSERT(weights->n_dimension == 2);
    /* MAKE SURE VECTOR/MATRIX MULTIPLICATION IS VALID */
    ADCI_ASSERT(input->shape[input->n_dimension - 1] == weights->shape[1]);
    /* COMPUTE OUTPUT SIZE */
    struct adci_output_format output_format = {.n_dimension = 2};
    output_format.shape[0] = input->shape[0];
    output_format.shape[1] = weights->shape[0];
    return output_format;
}
struct adci_reduce_max_format_info{
    struct adci_vector free_dim_mapping;
    struct adci_vector reduced_dim_mapping;
    struct adci_vector axis_view;
    unsigned int reduce_volume;
    bool keep_dims;
};
static struct adci_output_format adci_tensor_op_reduce_max_format(struct adci_tensor *tensor, struct adci_tensor *axis, struct adci_tensor *keepdims){
    ADCI_ASSERT(axis->n_dimension == 1);
    ADCI_ASSERT(axis->dtype == ADCI_I32);
    ADCI_ASSERT(axis->shape[0] <= tensor->n_dimension);
    struct adci_reduce_max_format_info *reduce_max_info = ADCI_ALLOC(sizeof(struct adci_reduce_max_format_info));
    reduce_max_info->keep_dims = false;
    if(keepdims){
        ADCI_ASSERT(keepdims->dtype == ADCI_I32);
        ADCI_ASSERT(keepdims->n_dimension == 1);
        reduce_max_info->keep_dims = adci_tensor_get_i32(keepdims, 0) != 0;
    }
    reduce_max_info->reduce_volume = 1;
    reduce_max_info->free_dim_mapping = adci_vector_init(sizeof(unsigned int));
    reduce_max_info->reduced_dim_mapping = adci_vector_init(sizeof(unsigned int));
    reduce_max_info->axis_view.data = axis->data;
    reduce_max_info->axis_view.bsize = adci_tensor_dtype_size(axis->dtype);
    reduce_max_info->axis_view.length = axis->shape[0];
    for(unsigned int i = 0; i < tensor->n_dimension; i++){
        if(!adci_vector_has(&reduce_max_info->axis_view, &i)){
            adci_vector_add(&reduce_max_info->free_dim_mapping, &i);
            continue;
        }
        /* PART OF THE REDUCED DIMS */
        adci_vector_add(&reduce_max_info->reduced_dim_mapping, &i);
        reduce_max_info->reduce_volume *= tensor->shape[i];
    }
    /* GIVE FINAL SIZE TO OUTPUT TENSOR */
    struct adci_output_format output_format = {.n_dimension = reduce_max_info->keep_dims ? tensor->n_dimension : reduce_max_info->free_dim_mapping.length, .data = reduce_max_info };
    memcpy(output_format.shape, tensor->shape, tensor->n_dimension * sizeof(unsigned int));
    for(unsigned int i = 0; i < reduce_max_info->free_dim_mapping.length; i++) 
        output_format.shape[i] = tensor->shape[((unsigned int *)reduce_max_info->free_dim_mapping.data)[i]];
    if(output_format.n_dimension == 0){
        output_format.n_dimension = 1;
        output_format.shape[0] = 1;
    }
    return output_format;
}

static struct adci_output_format adci_tensor_op_batch_matmult_format(struct adci_tensor *first, struct adci_tensor *second){
    ADCI_ASSERT(first->n_dimension == 3 && second->n_dimension == 3);
    /* SAME BATCH COUNT */
    ADCI_ASSERT(first->shape[0] == second->shape[0]);
    ADCI_ASSERT(first->shape[2] == second->shape[1]);
    struct adci_output_format output_format = {0};
    output_format.n_dimension = first->n_dimension;
    output_format.shape[0] = first->shape[0];
    output_format.shape[1] = first->shape[1];
    output_format.shape[2] = second->shape[2];
    return output_format;
}

static struct adci_output_format adci_tensor_argmax_format(struct adci_tensor *tensor, struct adci_tensor *dim, struct adci_tensor *keep_dim){
    struct adci_output_format output_format = {0};
    ADCI_ASSERT(dim->n_dimension == 1 && dim->shape[0] == 1 && dim->dtype == ADCI_I32);
    const uint32_t reduced_dim = adci_tensor_get_i32(dim, 0);
    ADCI_ASSERT(reduced_dim < tensor->n_dimension);
    ADCI_ASSERT(keep_dim == NULL || (keep_dim->n_dimension == 1 && keep_dim->shape[0] == 1));
    const bool keep_reduced_dim = (keep_dim != NULL) && (adci_tensor_get_i32(keep_dim, 0) == 1);
    memcpy(output_format.shape, tensor->shape, sizeof(output_format.shape));
    output_format.shape[reduced_dim] = 1;
    output_format.n_dimension = tensor->n_dimension;
    if(!keep_reduced_dim){
        output_format.n_dimension -= 1;
        for(unsigned int i = reduced_dim; i < tensor->n_dimension - 1; i++)
            output_format.shape[i] = output_format.shape[i + 1];
    }
    return output_format;
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
    struct adci_output_format output_format = adci_tensor_op_reshape_format(inputs);
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    const unsigned int tensor_size = adci_prepare_output_tensor(output_format.shape, output_format.n_dimension, tensor->dtype, output);
    if(tensor == output) return;
    memcpy(output->data, tensor->data, tensor_size);
}

void ADCI_EXIT_POINT adci_tensor_pad(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 2);
    struct adci_output_format output_format = adci_tensor_op_padding_format(inputs);
    struct adci_tensor *element = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    struct adci_tensor *padding = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    const unsigned int previous_count = adci_tensor_element_count(element);
    struct adci_tensor temp = *element;
    /* SO WE DONT CLEAR INPUT TENSOR BEFORE MOVING THE DATA */
    if(element == output) output->data = NULL;
    adci_prepare_output_tensor(output_format.shape, element->n_dimension, element->dtype, output);
    const unsigned int volume = adci_tensor_element_count(output);
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

void ADCI_EXIT_POINT adci_tensor_prelu_args(
    struct adci_tensor *element,
    struct adci_tensor *parameters, 
    struct adci_tensor *output)
    {
    /* f(y) = y if y >= 0 and f(y) = a * y if y < 0 */
    /* EACH CHANNEL HAS A DIFFERENT PARAMETER a */
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

void ADCI_EXIT_POINT adci_tensor_prelu(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 2);
    struct adci_tensor *element = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    struct adci_tensor *parameters = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    ADCI_ASSERT(parameters->n_dimension <= element->n_dimension);
    adci_tensor_prelu_args(element, parameters, output);
}

void ADCI_EXIT_POINT adci_tensor_relu_args(struct adci_tensor *tensor, struct adci_tensor *output){
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

void ADCI_EXIT_POINT adci_tensor_relu(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 1);
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    adci_tensor_relu_args(tensor, output);
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
    /* TODO, ADD SUPPORT FOR F64, F16 AND SPECIFY THE TYPE WITH AN INPUT TENSOR */
    enum adci_tensor_type output_type = ADCI_F32;
    const unsigned int dim = adci_tensor_get_i32(axis, 0);
    ADCI_ASSERT(dim < tensor->n_dimension);
    struct adci_tensor temp = *tensor;
    if(tensor == output) output->data = NULL;
    adci_prepare_output_tensor(tensor->shape, tensor->n_dimension, output_type, output);
    const unsigned int count = adci_tensor_element_count(&temp) / temp.shape[dim];
    const unsigned int in_elem_size = adci_tensor_dtype_size(temp.dtype);
    const unsigned int ou_elem_size = adci_tensor_dtype_size(output->dtype);
    const single_op_template_fn_t cast = single_cast_op_template_fns[temp.dtype * ADCI_NUM_SUPPORTED_TYPES + output->dtype]; 
    struct adci_tensor max;
    memset(&max, 0, sizeof(struct adci_tensor));
    struct adci_tensor *reduce_dims = adci_tensor_init_vargs(1, ADCI_I32, 1);
    adci_tensor_alloc_set(reduce_dims, (unsigned int[]){dim});
    adci_tensor_reduce_max_args(&temp, reduce_dims, NULL, &max);
    adci_tensor_free(reduce_dims);
    /* WILL CONTAIN THE TEMPORARY INPUT CASTED VALUES */
    float elements[temp.shape[dim]];
    struct adci_multi_dim_counter counter = adci_tensor_alldim_counter_except(&temp, dim);
    struct adci_multi_dim_counter reduced_counter = adci_tensor_alldim_counter(&temp);
    for(unsigned int i = 0; i < count; i++){
        const unsigned int current_offset = adci_tensor_get_counter_offset(counter);
        uint8_t current_max[ou_elem_size];
        cast(max.data + adci_tensor_get_counter_offset(reduced_counter) * in_elem_size, NULL, current_max);
        float sum = 0.f;
        for(unsigned int j = 0; j < temp.shape[dim]; j++){
            /* NO NEED FOR THIS STEP IF INPUT'S TYPE IS F32 */
            cast(temp.data + (current_offset + j * counter.precomputed_volumes[dim]) * in_elem_size, NULL, elements + j);
            /* TODO, CHANGE TYPE FROM HARDCODED FLOAT TO OTHER SUPPORTED TYPES */
            elements[j] = exp(elements[j] - ((float *)current_max)[0]);
            sum += elements[j];
        }
        for(unsigned int j = 0; j < temp.shape[dim]; j++)
            ((float *)output->data)[current_offset + j * counter.precomputed_volumes[dim]] = elements[j] / sum;
        adci_tensor_increase_counter(&counter);
        adci_tensor_increase_counter(&reduced_counter);
    }
    ADCI_FREE(max.data);
    if(tensor == output) ADCI_FREE(temp.data);
}

void ADCI_EXIT_POINT adci_tensor_reduce_max_args(
    struct adci_tensor *tensor,
    struct adci_tensor *axis, 
    struct adci_tensor *keepdims, 
    struct adci_tensor *output)
    {
    struct adci_tensor temp = *tensor;
    if(tensor == output) output->data = NULL;

    struct adci_output_format format = adci_tensor_op_reduce_max_format(tensor, axis, keepdims);
    const unsigned int element_size = adci_tensor_dtype_size(tensor->dtype);
    const unsigned int output_size = adci_prepare_output_tensor(format.shape, format.n_dimension, temp.dtype, output);
    const unsigned int out_volume = output_size / element_size;
    struct adci_reduce_max_format_info *reduce_max_info = format.data;

    int8_t maximum[element_size];
    struct adci_multi_dim_counter free_dim_counter = 
        adci_tensor_init_multidim_counter(&temp, (unsigned int *)reduce_max_info->free_dim_mapping.data, reduce_max_info->free_dim_mapping.length);
    for(unsigned int i = 0; i < out_volume; i++){
        unsigned int fixed_offset = 
            adci_tensor_get_counter_offset(free_dim_counter);
        adci_reset_value(maximum, tensor->dtype);
        struct adci_multi_dim_counter reduced_dim_counter = adci_tensor_init_multidim_counter(&temp, (unsigned int *)reduce_max_info->reduced_dim_mapping.data, reduce_max_info->reduced_dim_mapping.length);
        for(unsigned int j = 0; j < reduce_max_info->reduce_volume; j++){
            /* CHECK FOR MAXIMUM VALUE AMONGST THE REDUCED DIMENSIONS */
            unsigned int reduced_offset = adci_tensor_get_counter_offset(reduced_dim_counter);
            int8_t *current = (int8_t *)temp.data + (fixed_offset + reduced_offset) * element_size;
            adci_compare_max(current, maximum, temp.dtype);
            adci_tensor_increase_counter(&reduced_dim_counter);
        }
        memcpy((int8_t *)output->data + i * element_size, maximum, element_size);
        adci_tensor_increase_counter(&free_dim_counter);
    }
    if(reduce_max_info->keep_dims){
        output->n_dimension = temp.n_dimension;
        for(unsigned int i = 0; i < temp.n_dimension; i++){
            output->shape[i] = temp.shape[i];
            if(adci_vector_has(&reduce_max_info->axis_view, &i)) output->shape[i] = 1; 
        }
    }
    adci_vector_free(&reduce_max_info->free_dim_mapping);
    adci_vector_free(&reduce_max_info->reduced_dim_mapping);
    ADCI_FREE(reduce_max_info);
    if(tensor == output) ADCI_FREE(temp.data);
}

void ADCI_EXIT_POINT adci_tensor_reduce_max(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 2 || inputs.length == 3);
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    struct adci_tensor *axis = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    struct adci_tensor *keepdims = NULL;
    if(inputs.length == 3)
        keepdims = *(struct adci_tensor **)adci_vector_get(&inputs, 2);
    adci_tensor_reduce_max_args(tensor, axis, keepdims, output);
}

void ADCI_EXIT_POINT adci_tensor_concat(struct adci_vector inputs, struct adci_tensor *output){
    struct adci_output_format output_format = adci_tensor_op_concat_format(inputs);
    struct adci_tensor *first = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    struct adci_tensor temp = *output;
    bool output_in_input = first == output;
    for(unsigned int i = 1; i < inputs.length; i++){
        struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, i);
        if(tensor != output) continue;
        output_in_input = true;
        break;
    }
    if(output_in_input) output->data = NULL;
    adci_prepare_output_tensor(output_format.shape, output_format.n_dimension, first->dtype, output);
    const struct adci_tensor *axis = *(struct adci_tensor **)adci_vector_get(&inputs, inputs.length - 1); 
    const unsigned int dim = adci_tensor_get_i32(axis, 0);
    unsigned int count = adci_tensor_element_count_ext(dim, first->shape);
    const unsigned int element_size = adci_tensor_dtype_size(first->dtype);
    const unsigned int copy_block_size = adci_tensor_element_count_ext(first->n_dimension - dim - 1, first->shape + dim + 1) * element_size;
    const unsigned int output_single_copy_block_size = output_format.shape[dim] * copy_block_size;
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

void ADCI_EXIT_POINT adci_tensor_generic_pool2D_args(
    struct adci_tensor *tensor, 
    struct adci_tensor *size, 
    struct adci_tensor *stride,
    struct adci_tensor *dims,
    struct adci_tensor *output,
    adci_pool2D_op pool2D_op)
    {
    struct adci_output_format output_format = adci_tensor_op_pool2D_format(tensor, size, stride, dims);
    struct adci_tensor temp = *tensor;
    if(tensor == output) output->data = NULL;
    unsigned int output_volume = adci_prepare_output_tensor(output_format.shape, output_format.n_dimension, tensor->dtype, output) / adci_tensor_dtype_size(temp.dtype);
    /* WE WANT A COUNTER ARROUND THE FREE DIMENSIONS (DIMENSIONS NOT CONCERNED BY THE 2D POOL) */
    unsigned int index = 0;
    unsigned int free_dims[temp.n_dimension - 2];
    for(int i = 0; i < (int)temp.n_dimension; i++)
        if(adci_tensor_get_i32(dims, 0) != i && adci_tensor_get_i32(dims, 1) != i) free_dims[index++] = i;
    struct adci_multi_dim_counter output_counter = adci_tensor_init_multidim_counter(output, free_dims, dims->shape[0]);
    struct adci_multi_dim_counter tensor_counter = adci_tensor_init_multidim_counter(&temp, free_dims, dims->shape[0]);
    const unsigned int matrix_count = output_volume / (output->shape[adci_tensor_get_i32(dims, 0)] * output->shape[adci_tensor_get_i32(dims, 1)]);
    for(unsigned int i = 0; i < matrix_count; i++){
        for(unsigned int width = 0; width < output->shape[adci_tensor_get_i32(dims, 0)]; width++){
            for(unsigned int height = 0; height < output->shape[adci_tensor_get_i32(dims, 1)]; height++)
                pool2D_op(&temp, size, stride, dims, output, tensor_counter, output_counter, width, height);
        }
        /* PASS TO NEXT 2D PLANE */
        adci_tensor_increase_counter(&output_counter);
        adci_tensor_increase_counter(&tensor_counter);
    }
    if(tensor == output) ADCI_FREE(temp.data);
}

void ADCI_EXIT_POINT adci_tensor_max_pool2D_args(
    struct adci_tensor *tensor, 
    struct adci_tensor *size, 
    struct adci_tensor *stride,
    struct adci_tensor *dims,
    struct adci_tensor *output)
    {
    adci_tensor_generic_pool2D_args(tensor, size, stride, dims, output, adci_tensor_max_pool2D_op);
}

void ADCI_EXIT_POINT adci_tensor_avg_pool2D_args(
    struct adci_tensor *tensor, 
    struct adci_tensor *size, 
    struct adci_tensor *stride,
    struct adci_tensor *dims,
    struct adci_tensor *output)
    {
    adci_tensor_generic_pool2D_args(tensor, size, stride, dims, output, adci_tensor_avg_pool2D_op);
}

void ADCI_EXIT_POINT adci_tensor_avg_pool2D(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 4);
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0); 
    /* HEIGHT, WIDTH ORDER */
    struct adci_tensor *size   = *(struct adci_tensor **)adci_vector_get(&inputs, 1); 
    struct adci_tensor *stride = *(struct adci_tensor **)adci_vector_get(&inputs, 2);
    /* SPECIFIES THE 2 DIMENSIONS TO BE CONSIDERED FOR THE 2D POOL [HEIGHT_DIM, WIDTH_DIM]*/
    struct adci_tensor *dims   = *(struct adci_tensor **)adci_vector_get(&inputs, 3);
    adci_tensor_avg_pool2D_args(tensor, size, stride, dims, output);
}

void ADCI_EXIT_POINT adci_tensor_max_pool2D(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 4);
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0); 
    /* HEIGHT, WIDTH ORDER */
    struct adci_tensor *size   = *(struct adci_tensor **)adci_vector_get(&inputs, 1); 
    struct adci_tensor *stride = *(struct adci_tensor **)adci_vector_get(&inputs, 2);
    /* SPECIFIES THE 2 DIMENSIONS TO BE CONSIDERED FOR THE 2D POOL [HEIGHT_DIM, WIDTH_DIM]*/
    struct adci_tensor *dims   = *(struct adci_tensor **)adci_vector_get(&inputs, 3);
    adci_tensor_max_pool2D_args(tensor, size, stride, dims, output);
}

/*@stride: [HEIGHT, WIDTH] ORDER */
/*@dims: SPECIFIES THE INDECES OF DIMENSIONS TO BE CONSIDERED FOR THE 2D MASK [HEIGHT_DIM, WIDTH_DIM, CHANNEL_DIM] */

void ADCI_EXIT_POINT adci_tensor_conv2D_args(
    struct adci_tensor *tensor,
    struct adci_tensor *filter,
    struct adci_tensor *stride,
    struct adci_tensor *dims,
    struct adci_tensor *output)
    {
    struct adci_output_format output_format = adci_tensor_op_conv2D_format(tensor, filter, stride, dims);
    const unsigned int batch_dim = *(unsigned int *)output_format.data;
    ADCI_FREE(output_format.data);
    struct adci_tensor temp = *tensor;
    if(tensor == output) output->data = NULL;
    adci_prepare_output_tensor(output_format.shape, temp.n_dimension, temp.dtype, output);
    for(unsigned int i = 0; i < output->shape[batch_dim]; i++){
        for(unsigned int channel = 0; channel < filter->shape[0]; channel++){
            /* RUN CONV ON CURRENT 2D/3D VOLUME */
            adci_single_channel_conv(&temp, filter, stride, dims, output, i, batch_dim, channel);
        }
    }
    if(tensor == output) ADCI_FREE(temp.data);
}

/* FILTER TENSOR HAS TO BE IN THE SHAPE [OUT_CHANNEL, HEIGHT, WIDTH, IN_CHANNEL]*/
void ADCI_EXIT_POINT adci_tensor_conv2D(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 4);
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0); 
    struct adci_tensor *filter = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
    struct adci_tensor *stride = *(struct adci_tensor **)adci_vector_get(&inputs, 2);
    struct adci_tensor *dims = *(struct adci_tensor **)adci_vector_get(&inputs, 3);
    adci_tensor_conv2D_args(tensor, filter, stride, dims, output);
}

void ADCI_EXIT_POINT adci_tensor_transpose_args(struct adci_tensor *tensor, struct adci_tensor *dims, struct adci_tensor *output){
    struct adci_output_format output_format = adci_tensor_op_transpose_format(tensor, dims);
    struct adci_tensor temp = *tensor;
    if(tensor == output) output->data = NULL;
    const unsigned int element_size = adci_tensor_dtype_size(tensor->dtype);
    const unsigned int volume = adci_prepare_output_tensor(output_format.shape, tensor->n_dimension, tensor->dtype, output) / element_size;
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
    ADCI_ASSERT(inputs.length == 2);
    struct adci_tensor *input = *(struct adci_tensor **)adci_vector_get(&inputs, 0); 
    struct adci_tensor *weights = *(struct adci_tensor **)adci_vector_get(&inputs, 1); 
    struct adci_output_format output_format = adci_tensor_op_fully_connected_format(input, weights);
    struct adci_tensor temp = *input;
    if(input == output) output->data = NULL;
    adci_prepare_output_tensor(output_format.shape, output_format.n_dimension, temp.dtype, output);
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

void ADCI_EXIT_POINT adci_tensor_batch_matmult_args(
    struct adci_tensor *first, 
    struct adci_tensor *second, 
    struct adci_tensor *output)
    {
    struct adci_output_format output_format = adci_tensor_op_batch_matmult_format(first, second);
    struct adci_tensor first_cp = *first;
    struct adci_tensor second_cp = *second;
    void *data_copy = NULL;
    if(first == output || second == output){
        data_copy = output->data;
        output->data = NULL;
    } 
    const unsigned int element_size = adci_tensor_dtype_size(first->dtype);
    const unsigned int second_element_size = adci_tensor_dtype_size(second->dtype);
    const single_op_template_fn_t mult_op = single_mult_op_template_fns[first->dtype * ADCI_NUM_SUPPORTED_TYPES + second->dtype];
    const single_op_template_fn_t add_op = single_add_op_template_fns[first->dtype * ADCI_NUM_SUPPORTED_TYPES + first->dtype];
    adci_prepare_output_tensor(output_format.shape, output_format.n_dimension, first->dtype, output);
    for(unsigned int batch_idx = 0; batch_idx < first_cp.shape[0]; batch_idx++){
        /* RUN MATMUL ON CURRENT MATRICES */
        const unsigned int bfoffset = batch_idx * first_cp.shape[1] * first_cp.shape[2];
        const unsigned int bsoffset = batch_idx * second_cp.shape[1] * second_cp.shape[2];
        for(unsigned int i = 0; i < first_cp.shape[1]; i++){
            const unsigned int cfoffset = i * first_cp.shape[2];
            for(unsigned int j = 0; j < second_cp.shape[2]; j++){
                const unsigned int csoffset = j;
                /* RUN DOT PRODUCT */
                uint8_t container[element_size];
                adci_reset_value(container, first_cp.dtype);
                uint8_t temp[element_size];
                for(unsigned int k = 0; k < first_cp.shape[2]; k++){
                    const unsigned int ffoffset = (bfoffset + cfoffset + k) * element_size;
                    const unsigned int sfoffset = (bsoffset + csoffset + k * second_cp.shape[2]) * second_element_size;
                    mult_op((uint8_t *)first_cp.data + ffoffset, (uint8_t *)second_cp.data + sfoffset, temp);
                    add_op(temp, container, container);
                }
                /* WRITE RESULT TO OUTPUT TENSOR */
                const unsigned int output_offset = (i * output->shape[2] + j) * element_size;
                memcpy((uint8_t *)output->data + output_offset, container, element_size); 
            }
        }
    }
    if(data_copy) ADCI_FREE(data_copy);
}

void ADCI_EXIT_POINT adci_tensor_batch_matmult(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 2);
    struct adci_tensor *first = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    struct adci_tensor *second = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    adci_tensor_batch_matmult_args(first, second, output);
}

void ADCI_EXIT_POINT adci_tensor_argmax_args(struct adci_tensor *tensor, struct adci_tensor *dim, struct adci_tensor *keep_dim, struct adci_tensor *output){
    struct adci_output_format output_format = adci_tensor_argmax_format(tensor, dim, keep_dim);
    const uint32_t reduced_dim = adci_tensor_get_i32(dim, 0);
    const unsigned int count = adci_tensor_element_count(tensor) / tensor->shape[reduced_dim];
    struct adci_multi_dim_counter counter = adci_tensor_alldim_counter_except(tensor, reduced_dim);
    const unsigned int element_size = adci_tensor_dtype_size(tensor->dtype);
    single_op_template_fn_t max_op = single_max_op_template_fns[tensor->dtype * ADCI_NUM_SUPPORTED_TYPES + tensor->dtype];
    struct adci_tensor temp = *tensor;
    if(tensor == output) output->data = NULL;
    adci_prepare_output_tensor(output_format.shape, output_format.n_dimension, ADCI_I32, output);
    struct adci_multi_dim_counter output_counter = adci_tensor_alldim_counter(output);
    const unsigned int output_element_size = adci_tensor_dtype_size(ADCI_I32);
    for(unsigned int i = 0; i < count; i++){
        /* FIND THE MAXIMUM IN REDUCED DIM FOR CURRENT CONFIG */
        const unsigned int offset = adci_tensor_get_counter_offset(counter);
        uint32_t index = 0;
        uint8_t container[element_size];
        memcpy(container, tensor->data + offset * element_size, element_size);
        for(unsigned int j = 1; j < tensor->shape[reduced_dim]; j++){
            const unsigned int foffset = offset + j * counter.precomputed_volumes[reduced_dim];
            uint8_t temp[element_size];
            memcpy(temp, container, element_size);
            max_op((uint8_t *)tensor->data + foffset * element_size, container, container);
            if(memcmp(temp, container, element_size) != 0) index = j;
        }
        /* WRITE TO OUTPUT TENSOR THE INDEX */
        const unsigned int output_offset = adci_tensor_get_counter_offset(output_counter);
        memcpy((uint8_t *)output->data + output_offset * output_element_size, &index, output_element_size);
        adci_tensor_increase_counter(&output_counter);
        adci_tensor_increase_counter(&counter);
    }
    if(tensor == output) ADCI_FREE(temp.data);
}

void ADCI_EXIT_POINT adci_tensor_argmax(struct adci_vector inputs, struct adci_tensor *output){
    ADCI_ASSERT(inputs.length == 2 || inputs.length == 3);
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0); 
    struct adci_tensor *dim = *(struct adci_tensor **)adci_vector_get(&inputs, 1); 
    struct adci_tensor *keep_dim = NULL;
    if(inputs.length == 3)
        keep_dim = *(struct adci_tensor **)adci_vector_get(&inputs, 2); 
    adci_tensor_argmax_args(tensor, dim, keep_dim, output);
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
    case ADCI_TENSOR_MAX_POOL2D: return adci_tensor_max_pool2D(inputs, output);
    case ADCI_TENSOR_AVG_POOL2D: return adci_tensor_avg_pool2D(inputs, output);
    case ADCI_TENSOR_CONV2D: return adci_tensor_conv2D(inputs, output);
    case ADCI_TENSOR_TRANSPOSE: return adci_tensor_transpose(inputs, output);
    case ADCI_TENSOR_FULLY_CONNECTED: return adci_tensor_fully_connected(inputs, output);
    case ADCI_TENSOR_BATCH_MATMUL: return adci_tensor_batch_matmult(inputs, output);
    default:
        ADCI_LOG(ADCI_ERROR, "OPERATION: %s NOT IMPLEMENTED YET", adci_tensor_op_str(op));
        ADCI_ASSERT(false);
    }
}

void ADCI_EXIT_POINT adci_tensor_compute_op_shape(struct adci_vector inputs, struct adci_tensor *output, enum adci_tensor_op op){
    /* NO INPUTS IN THIS CASE */
    if(op == ADCI_TENSOR_INPUT) return;
    struct adci_tensor *tensor = *(struct adci_tensor **)adci_vector_get(&inputs, 0);
    switch(op){
    case ADCI_TENSOR_ADD:
    case ADCI_TENSOR_SUB:
    case ADCI_TENSOR_COPY:
    case ADCI_TENSOR_PRELU:
    case ADCI_TENSOR_RELU:
    case ADCI_TENSOR_CAST:
    case ADCI_TENSOR_SOFTMAX:
    case ADCI_TENSOR_MUL:
        output->n_dimension = tensor->n_dimension;
        memcpy(output->shape, tensor->shape, sizeof(tensor->shape));
        break;
    case ADCI_TENSOR_RESHAPE:{
        struct adci_output_format output_format = adci_tensor_op_reshape_format(inputs);
        output->n_dimension = output_format.n_dimension;
        memcpy(output->shape, output_format.shape, output_format.n_dimension * sizeof(uint32_t));
    }break;
    case ADCI_TENSOR_PAD:{
        struct adci_output_format output_format = adci_tensor_op_padding_format(inputs);
        output->n_dimension = output_format.n_dimension;
        memcpy(output->shape, output_format.shape, output_format.n_dimension * sizeof(uint32_t));
    }break;
    case ADCI_TENSOR_REDUCE_MAX:{
        struct adci_tensor *axis = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
        struct adci_tensor *keep_dims = NULL; 
        if(inputs.length == 3) keep_dims = *(struct adci_tensor **)adci_vector_get(&inputs, 2);
        struct adci_output_format output_format = adci_tensor_op_reduce_max_format(tensor, axis, keep_dims);
        output->n_dimension = output_format.n_dimension;
        memcpy(output->shape, output_format.shape, output_format.n_dimension * sizeof(uint32_t));
        ADCI_FREE(output_format.data);
    }break;
    case ADCI_TENSOR_MAX_POOL2D:
    case ADCI_TENSOR_AVG_POOL2D:{
        ADCI_ASSERT(inputs.length == 4);
        struct adci_tensor *size   = *(struct adci_tensor **)adci_vector_get(&inputs, 1); 
        struct adci_tensor *stride = *(struct adci_tensor **)adci_vector_get(&inputs, 2);
        struct adci_tensor *dims   = *(struct adci_tensor **)adci_vector_get(&inputs, 3);
        struct adci_output_format output_format = adci_tensor_op_pool2D_format(tensor, size, stride, dims);
        output->n_dimension = output_format.n_dimension;
        memcpy(output->shape, output_format.shape, output_format.n_dimension * sizeof(uint32_t));
    }break;
    case ADCI_TENSOR_CONCAT:{
        struct adci_output_format output_format = adci_tensor_op_concat_format(inputs);
        output->n_dimension = output_format.n_dimension;
        memcpy(output->shape, output_format.shape, output_format.n_dimension * sizeof(uint32_t));
    }break;
    case ADCI_TENSOR_CONV2D:{
        struct adci_tensor *filter = *(struct adci_tensor **)adci_vector_get(&inputs, 1); 
        struct adci_tensor *stride = *(struct adci_tensor **)adci_vector_get(&inputs, 2);
        struct adci_tensor *dims   = *(struct adci_tensor **)adci_vector_get(&inputs, 3);
        struct adci_output_format output_format = adci_tensor_op_conv2D_format(tensor, filter, stride, dims);
        ADCI_FREE(output_format.data);
        output->n_dimension = output_format.n_dimension;
        memcpy(output->shape, output_format.shape, output_format.n_dimension * sizeof(uint32_t));
    }break;
    case ADCI_TENSOR_TRANSPOSE:{
        struct adci_tensor *dims = *(struct adci_tensor **)adci_vector_get(&inputs, 1); 
        struct adci_output_format output_format = adci_tensor_op_transpose_format(tensor, dims);
        output->n_dimension = output_format.n_dimension;
        memcpy(output->shape, output_format.shape, output_format.n_dimension * sizeof(uint32_t));
    }break;
    case ADCI_TENSOR_FULLY_CONNECTED:{
        struct adci_tensor *weights = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
        struct adci_output_format output_format = adci_tensor_op_fully_connected_format(tensor, weights);
        output->n_dimension = output_format.n_dimension;
        memcpy(output->shape, output_format.shape, output_format.n_dimension * sizeof(uint32_t));
    }break;
    case ADCI_TENSOR_BATCH_MATMUL:{
        struct adci_tensor *second = *(struct adci_tensor **)adci_vector_get(&inputs, 1);
        struct adci_output_format output_format = adci_tensor_op_batch_matmult_format(tensor, second);
        output->n_dimension = output_format.n_dimension;
        memcpy(output->shape, output_format.shape, output_format.n_dimension * sizeof(uint32_t));
    }break;
    case ADCI_TENSOR_TRANSPOSE_CONV:
    default:
        ADCI_ASSERT("OUTPUT SIZE FUNCTIONS NOT IMPLEMENTED FOR OP" == 0);
    }
}

const char * adci_tensor_op_str(enum adci_tensor_op op){
    #define OP_STR_CASE(_op) case _op: return ADCI_TOKEN2STR(_op)
    switch (op){
        OP_STR_CASE(ADCI_TENSOR_INPUT);
        OP_STR_CASE(ADCI_TENSOR_COPY);
        OP_STR_CASE(ADCI_TENSOR_ADD);
        OP_STR_CASE(ADCI_TENSOR_SUB);
        OP_STR_CASE(ADCI_TENSOR_RESHAPE);
        OP_STR_CASE(ADCI_TENSOR_PAD);
        OP_STR_CASE(ADCI_TENSOR_PRELU);
        OP_STR_CASE(ADCI_TENSOR_CAST);
        OP_STR_CASE(ADCI_TENSOR_SOFTMAX);
        OP_STR_CASE(ADCI_TENSOR_REDUCE_MAX);
        OP_STR_CASE(ADCI_TENSOR_CONCAT);
        OP_STR_CASE(ADCI_TENSOR_MUL);
        OP_STR_CASE(ADCI_TENSOR_MAX_POOL2D);
        OP_STR_CASE(ADCI_TENSOR_RELU);
        OP_STR_CASE(ADCI_TENSOR_CONV2D);
        OP_STR_CASE(ADCI_TENSOR_TRANSPOSE);
        OP_STR_CASE(ADCI_TENSOR_FULLY_CONNECTED);
        OP_STR_CASE(ADCI_TENSOR_BATCH_MATMUL);
        OP_STR_CASE(ADCI_TENSOR_AVG_POOL2D);

        OP_STR_CASE(ADCI_TENSOR_TRANSPOSE_CONV);
        default: return "INVALID OP";
    }
}
    