#include "gtest/gtest.h"

extern "C"{
#include "adci_tensor.h"
#include "adci_tensor_op.h"
}

#define ADCI_TENSOR_OP_SUITE_NAME ADCI_TENSOR_OP

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_add){
    unsigned int shape[] = {10, 15};
    adci_tensor *first = adci_tensor_init(2, shape, ADCI_F32);
    adci_tensor *second = adci_tensor_init(2, shape, ADCI_F32);
    adci_tensor *output = adci_tensor_init(2, shape, ADCI_F32);
    adci_tensor_alloc(first);
    adci_tensor_alloc(second);
    adci_tensor_alloc(output);
    const unsigned int volume = 10 * 15;
    for(unsigned int i = 0; i < volume; i++){
       ((float *)first->data)[i] = static_cast<float>(i);
       ((float *)second->data)[i] = static_cast<float>(volume - i);           
    }
    adci_tensor * input_arr[] = {first, second};
    adci_vector inputs = adci_vector_from_array(input_arr, 2, sizeof(adci_tensor *));
    adci_tensor_add(inputs, output);
    adci_vector_free(&inputs);
    for(unsigned int i = 0; i < volume; i++){
        EXPECT_FLOAT_EQ(((float *)first->data)[i], static_cast<float>(i));
        EXPECT_FLOAT_EQ(((float *)second->data)[i], static_cast<float>(volume - i));
        EXPECT_FLOAT_EQ(((float *)output->data)[i], static_cast<float>(volume));
    }
    adci_tensor_free(output);
    adci_tensor_free(first);
    adci_tensor_free(second);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_sub){
    unsigned int shape[] = {10, 15};
    adci_tensor *first = adci_tensor_init(2, shape, ADCI_F32);
    adci_tensor *second = adci_tensor_init(2, shape, ADCI_F32);
    adci_tensor *output = adci_tensor_init(2, shape, ADCI_F32);
    adci_tensor_alloc(first);
    adci_tensor_alloc(second);
    adci_tensor_alloc(output);
    const unsigned int volume = 10 * 15;
    for(unsigned int i = 0; i < volume; i++){
       ((float *)first->data)[i] = static_cast<float>(i);
       ((float *)second->data)[i] = static_cast<float>(volume - i);           
    }
    adci_tensor * input_arr[] = {first, second};
    adci_vector inputs = adci_vector_from_array(input_arr, 2, sizeof(adci_tensor *));
    adci_tensor_sub(inputs, output);
    adci_vector_free(&inputs);
    for(unsigned int i = 0; i < volume; i++){
        EXPECT_FLOAT_EQ(((float *)first->data)[i], static_cast<float>(i));
        EXPECT_FLOAT_EQ(((float *)second->data)[i], static_cast<float>(volume - i));
        EXPECT_FLOAT_EQ(((float *)output->data)[i], static_cast<float>(2.f * i - volume));
    }
    adci_tensor_free(output);
    adci_tensor_free(first);
    adci_tensor_free(second);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_reshape_no_copy){
    unsigned int shape[] = {10, 15};
    unsigned int reshape[] = {1, 15, 10};
    unsigned int n_dims = sizeof(reshape) / sizeof(unsigned int);
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor *shape_t = adci_tensor_init_vargs(1, ADCI_I32, n_dims);
    adci_tensor_alloc_set(shape_t, reshape);
    adci_tensor *input_arr[] = {first, shape_t};
    adci_vector inputs = adci_vector_from_array(input_arr, 2, sizeof(adci_tensor *));
    adci_tensor_reshape(inputs, first);
    EXPECT_EQ(first->n_dimension, n_dims);
    for(unsigned int i = 0; i < n_dims; i++) EXPECT_EQ(first->shape[i], reshape[i]);
    adci_vector_free(&inputs);
    adci_tensor_free(first);
    adci_tensor_free(shape_t);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_reshape){
    unsigned int shape[] = {10, 15};
    unsigned int reshape[] = {2, 15, 5};
    unsigned int n_dims = sizeof(reshape) / sizeof(unsigned int);
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor_alloc(first);
    adci_tensor *shape_t = adci_tensor_init_vargs(1, ADCI_I32, n_dims);
    adci_tensor_alloc_set(shape_t, reshape);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++) 
        ((float *)first->data)[i] = (float)rand() / RAND_MAX;
    adci_tensor *output = adci_tensor_init(3, reshape, ADCI_F32);
    adci_tensor *input_arr[] = {first, shape_t};
    adci_vector inputs = adci_vector_from_array(input_arr, 2, sizeof(adci_tensor *));
    adci_tensor_reshape(inputs, output);
    EXPECT_EQ(first->n_dimension, sizeof(shape) / sizeof(unsigned int));
    EXPECT_EQ(output->n_dimension, n_dims);
    for(unsigned int i = 0; i < n_dims; i++) EXPECT_EQ(output->shape[i], reshape[i]);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++)
        EXPECT_FLOAT_EQ(((float *)first->data)[i], ((float *)output->data)[i]);
    adci_vector_free(&inputs);
    adci_tensor_free(output);
    adci_tensor_free(first);
    adci_tensor_free(shape_t);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_copy){
    unsigned int shape[] = {10, 15};
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor_alloc(first);
    adci_tensor *second = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor_alloc(second);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++) 
        ((float *)first->data)[i] = (float)rand() / RAND_MAX;
    ((float *)first->data)[0]  = 11.f;
    ((float *)second->data)[0] = 10.f;
    EXPECT_NE(((float *)first->data)[0], ((float *)second->data)[0]);
    adci_tensor_copy(first, second);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++) 
        EXPECT_FLOAT_EQ(((float *)first->data)[i], ((float *)second->data)[i]);
    adci_tensor_free(second);
    adci_tensor_free(first);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_pad_diff_output_first_dim){
    unsigned int shape[] = {4, 3};
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor *padding = adci_tensor_init_vargs(2, ADCI_I32, 2, 2);
    adci_tensor_alloc(padding);
    adci_tensor_set_i32(padding, 1, 0, 0);
    adci_tensor_set_i32(padding, 1, 0, 1);
    adci_tensor_set_i32(padding, 0, 1, 0);
    adci_tensor_set_i32(padding, 0, 1, 1);
    /* SHOULD PAD BY ONE ON EACH SIDE OF FIRST DIM ONLY AND FILL WITH ZEROS */
    adci_tensor_alloc(first);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++)
        ((float *)first->data)[i] = 1.f;
    adci_tensor *output = adci_tensor_init_vargs(2, ADCI_F32, 1, 1);
    struct adci_vector inputs = adci_vector_init(sizeof(adci_tensor *));
    adci_vector_add(&inputs, &first);
    adci_vector_add(&inputs, &padding);
    adci_tensor_pad(inputs, output);
    EXPECT_EQ(first->n_dimension, 2);
    EXPECT_EQ(first->shape[0], shape[0]);
    EXPECT_EQ(first->shape[1], shape[1]);
    EXPECT_EQ(output->n_dimension, 2);
    EXPECT_EQ(output->shape[0], shape[0] + 2);
    EXPECT_EQ(output->shape[1], shape[1]);
    for(unsigned int i = 0; i < output->shape[0] * output->shape[1]; i++){
        if(i < output->shape[1] || i >= (output->shape[0] - 1) * output->shape[1])
            EXPECT_FLOAT_EQ(((float *)output->data)[i], 0.f);
        else EXPECT_FLOAT_EQ(((float *)output->data)[i], 1.f);
    }
    adci_vector_free(&inputs);
    adci_tensor_free(first);
    adci_tensor_free(padding);
    adci_tensor_free(output);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_pad_diff_output_second_dim){
    unsigned int shape[] = {4, 3};
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor *padding = adci_tensor_init_vargs(2, ADCI_I32, 2, 2);
    adci_tensor_alloc(padding);
    adci_tensor_set_i32(padding, 0, 0, 0);
    adci_tensor_set_i32(padding, 0, 0, 1);
    adci_tensor_set_i32(padding, 1, 1, 0);
    adci_tensor_set_i32(padding, 1, 1, 1);
    adci_tensor_alloc(first);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++)
        ((float *)first->data)[i] = 1.f;
    adci_tensor *output = adci_tensor_init_vargs(2, ADCI_F32, 1, 1);
    struct adci_vector inputs = adci_vector_init(sizeof(adci_tensor *));
    adci_vector_add(&inputs, &first);
    adci_vector_add(&inputs, &padding);
    adci_tensor_pad(inputs, output);
    EXPECT_EQ(first->n_dimension, 2);
    EXPECT_EQ(first->shape[0], shape[0]);
    EXPECT_EQ(first->shape[1], shape[1]);
    EXPECT_EQ(output->n_dimension, 2);
    EXPECT_EQ(output->shape[0], shape[0]);
    EXPECT_EQ(output->shape[1], shape[1] + 2);
    for(unsigned int i = 0; i < output->shape[0] * output->shape[1]; i++){
        if(i % output->shape[1] == 0 || i % output->shape[1] == output->shape[1] - 1)
            EXPECT_FLOAT_EQ(((float *)output->data)[i], 0.f);
        else EXPECT_FLOAT_EQ(((float *)output->data)[i], 1.f);
    }
    adci_vector_free(&inputs);
    adci_tensor_free(first);
    adci_tensor_free(padding);
    adci_tensor_free(output);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_pad_diff_output_multiple_dim){
    unsigned int shape[] = {4, 3};
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor *padding = adci_tensor_init_vargs(2, ADCI_I32, 2, 2);
    adci_tensor_alloc(padding);
    adci_tensor_set_i32(padding, 1, 0, 0);
    adci_tensor_set_i32(padding, 0, 0, 1);
    adci_tensor_set_i32(padding, 1, 1, 0);
    adci_tensor_set_i32(padding, 2, 1, 1);
    adci_tensor_alloc(first);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++)
        ((float *)first->data)[i] = 1.f;
    adci_tensor *output = adci_tensor_init_vargs(2, ADCI_F32, 1, 1);
    struct adci_vector inputs = adci_vector_init(sizeof(adci_tensor *));
    adci_vector_add(&inputs, &first);
    adci_vector_add(&inputs, &padding);
    adci_tensor_pad(inputs, output);
    EXPECT_EQ(first->n_dimension, 2);
    EXPECT_EQ(first->shape[0], shape[0]);
    EXPECT_EQ(first->shape[1], shape[1]);
    EXPECT_EQ(output->n_dimension, 2);
    EXPECT_EQ(output->shape[0], shape[0] + 1);
    EXPECT_EQ(output->shape[1], shape[1] + 3);
    for(unsigned int i = 0; i < output->shape[0] * output->shape[1]; i++){
        const bool second_dim = i % output->shape[1] == 0 || i % output->shape[1] >= output->shape[1] - 2;
        const bool first_dim = i < output->shape[1];
        if(second_dim || first_dim)
            EXPECT_FLOAT_EQ(((float *)output->data)[i], 0.f);
        else EXPECT_FLOAT_EQ(((float *)output->data)[i], 1.f);
    }
    adci_vector_free(&inputs);
    adci_tensor_free(first);
    adci_tensor_free(padding);
    adci_tensor_free(output);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_prelu){
    unsigned int shape[] = {4, 3};
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor *output = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor *params = adci_tensor_init_vargs(1, ADCI_F32, shape[1]);
    adci_tensor_alloc(first);
    adci_tensor_alloc(params);
    adci_tensor_alloc(output);
    for(unsigned int i = 0; i < shape[1]; i++) adci_tensor_set_f32(params, static_cast<float>(rand()) / RAND_MAX, i);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++){
        float mult = i % 2 == 0 ? -1.f : 1.f;
        ((float *)first->data)[i] = mult * i;
    }
    struct adci_vector inputs = adci_vector_init(sizeof(adci_tensor *));
    adci_vector_add(&inputs, &first);
    adci_vector_add(&inputs, &params);
    adci_tensor_prelu(inputs, output);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++){
        float value = (i % 2 == 0 ? -1.f : 1.f) * i;
        if(value > 0) EXPECT_FLOAT_EQ(((float *)output->data)[i], static_cast<float>(i));
        else EXPECT_FLOAT_EQ(((float *)output->data)[i], ((float *)first->data)[i] * adci_tensor_get_f32(params, i % params->shape[0]));
    }
    adci_vector_free(&inputs);
    adci_tensor_free(first);
    adci_tensor_free(params);
    adci_tensor_free(output);
}   

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_relu){
    unsigned int shape[] = {4, 3};
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor_alloc(first);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++){
        float mult = i % 2 == 0 ? -1.f : 1.f;
        ((float *)first->data)[i] = mult * i;
    }
    struct adci_vector inputs = adci_vector_init(sizeof(adci_tensor *));
    adci_vector_add(&inputs, &first);
    adci_tensor output;
    memset(&output, 0, sizeof(adci_tensor));
    adci_tensor_relu(inputs, &output);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++){
        float value = (i % 2 == 0 ? -1.f : 1.f) * i;
        if(value > 0) EXPECT_FLOAT_EQ(((float *)output.data)[i], static_cast<float>(i));
        else EXPECT_FLOAT_EQ(((float *)output.data)[i], static_cast<float>(0));
    }
    adci_vector_free(&inputs);
    adci_tensor_free(first);
    ADCI_FREE(output.data);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_cast){
    unsigned int shape[] = {4, 3};
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_I32, shape[0], shape[1]);
    adci_tensor_alloc(first);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++)
        ((int32_t *)first->data)[i] = i;
    adci_tensor *dtype = adci_tensor_init_vargs(1, ADCI_I32, 1);
    adci_tensor_alloc(dtype);
    adci_tensor_set_i32(dtype, ADCI_F32, 0);
    adci_tensor output;
    memset(&output, 0, sizeof(adci_tensor));
    adci_vector inputs = adci_vector_init(sizeof(adci_tensor*));
    adci_vector_add(&inputs, &first);
    adci_vector_add(&inputs, &dtype);
    adci_tensor_cast(inputs, &output);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++){
        ASSERT_EQ(((int32_t *)first->data)[i], i);
        ASSERT_FLOAT_EQ(((float *)output.data)[i], static_cast<float>(i));
        if(i != 0){ ASSERT_NE(((int32_t *)first->data)[i], ((int32_t *)output.data)[i]); }
    }
    adci_vector_free(&inputs);
    adci_tensor_free(dtype);
    adci_tensor_free(first);
    ADCI_FREE(output.data);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_cast_f32_f32){
    unsigned int shape[] = {4, 3};
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor_alloc(first);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++)
        ((float *)first->data)[i] = static_cast<float>(i);
    adci_tensor *dtype = adci_tensor_init_vargs(1, ADCI_I32, 1);
    adci_tensor_alloc(dtype);
    adci_tensor_set_i32(dtype, ADCI_F32, 0);
    adci_tensor *output = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_vector inputs = adci_vector_init(sizeof(adci_tensor*));
    adci_vector_add(&inputs, &first);
    adci_vector_add(&inputs, &dtype);
    adci_tensor_cast(inputs, output);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++)
        ASSERT_FLOAT_EQ(((float *)output->data)[i], static_cast<float>(i));
    adci_vector_free(&inputs);
    adci_tensor_free(first);
    adci_tensor_free(dtype);
    adci_tensor_free(output);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_softmax_dim1){
    unsigned int shape[] = {2, 3};
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor_alloc(first);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++)
        ((float *)first->data)[i] = static_cast<float>(i % shape[1]);
    adci_tensor *dim = adci_tensor_init_vargs(1, ADCI_I32, 1);
    adci_tensor_alloc(dim);
    ((int32_t *)dim->data)[0] = 1;
    adci_tensor *output = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_vector inputs = adci_vector_init(sizeof(adci_tensor*));
    adci_vector_add(&inputs, &first);
    adci_vector_add(&inputs, &dim);
    adci_tensor_softmax(inputs, output);
    EXPECT_EQ(output->shape[0], shape[0]);    
    EXPECT_EQ(output->shape[1], shape[1]);
    float outputs[] = {0.09003057, 0.24472847, 0.66524096};
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++)
        EXPECT_FLOAT_EQ(((float *)output->data)[i], outputs[i % shape[1]]);
    adci_vector_free(&inputs);
    adci_tensor_free(first);
    adci_tensor_free(dim);
    adci_tensor_free(output);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_softmax_dim0){
    unsigned int shape[] = {2, 3};
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor_alloc(first);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++)
        ((float *)first->data)[i] = static_cast<float>(i % shape[1]);
    adci_tensor *dim = adci_tensor_init_vargs(1, ADCI_I32, 1);
    adci_tensor_alloc(dim);
    ((int32_t *)dim->data)[0] = 0;
    adci_tensor *output = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_vector inputs = adci_vector_init(sizeof(adci_tensor*));
    adci_vector_add(&inputs, &first);
    adci_vector_add(&inputs, &dim);
    adci_tensor_softmax(inputs, output);
    EXPECT_EQ(output->shape[0], shape[0]);    
    EXPECT_EQ(output->shape[1], shape[1]);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++)
        EXPECT_FLOAT_EQ(((float *)output->data)[i], 0.5f);
    adci_vector_free(&inputs);
    adci_tensor_free(first);
    adci_tensor_free(dim);
    adci_tensor_free(output);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_reduce_max_dim0){
    unsigned int shape[] = {2, 3};
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor_alloc(first);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++)
        ((float *)first->data)[i] = static_cast<float>(i);
    adci_tensor *dim = adci_tensor_init_vargs(1, ADCI_I32, 1);
    adci_tensor_alloc(dim);
    ((int32_t *)dim->data)[0] = 0;
    adci_tensor *output = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_vector inputs = adci_vector_init(sizeof(adci_tensor*));
    adci_vector_add(&inputs, &first);
    adci_vector_add(&inputs, &dim);
    adci_tensor_reduce_max(inputs, output);
    EXPECT_EQ(output->n_dimension, 1);    
    EXPECT_EQ(output->shape[0], shape[1]);    
    for(unsigned int i = 0; i < shape[1]; i++)
        EXPECT_FLOAT_EQ(((float *)output->data)[i], static_cast<float>(shape[1] + i));
    adci_vector_free(&inputs);
    adci_tensor_free(first);
    adci_tensor_free(dim);
    adci_tensor_free(output);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_reduce_max_dim1){
    unsigned int shape[] = {2, 3};
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor_alloc(first);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++)
        ((float *)first->data)[i] = static_cast<float>(i);
    adci_tensor *dim = adci_tensor_init_vargs(1, ADCI_I32, 1);
    adci_tensor_alloc(dim);
    ((int32_t *)dim->data)[0] = 1;
    adci_tensor *output = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_vector inputs = adci_vector_init(sizeof(adci_tensor*));
    adci_vector_add(&inputs, &first);
    adci_vector_add(&inputs, &dim);
    adci_tensor_reduce_max(inputs, output);
    EXPECT_EQ(output->n_dimension, 1);    
    EXPECT_EQ(output->shape[0], shape[0]);    
    EXPECT_FLOAT_EQ(((float *)output->data)[0], static_cast<float>(2));
    EXPECT_FLOAT_EQ(((float *)output->data)[1], static_cast<float>(5));
    adci_vector_free(&inputs);
    adci_tensor_free(first);
    adci_tensor_free(dim);
    adci_tensor_free(output);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_reduce_max_all){
    unsigned int shape[] = {2, 3};
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor_alloc(first);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++)
        ((float *)first->data)[i] = static_cast<float>(i);
    adci_tensor *dim = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc(dim);
    ((int32_t *)dim->data)[0] = 0;
    ((int32_t *)dim->data)[1] = 1;
    adci_tensor *output = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_vector inputs = adci_vector_init(sizeof(adci_tensor*));
    adci_vector_add(&inputs, &first);
    adci_vector_add(&inputs, &dim);
    adci_tensor_reduce_max(inputs, output);
    EXPECT_EQ(output->n_dimension, 1);    
    EXPECT_EQ(output->shape[0], 1);    
    EXPECT_FLOAT_EQ(((float *)output->data)[0], static_cast<float>(5));
    adci_vector_free(&inputs);
    adci_tensor_free(first);
    adci_tensor_free(dim);
    adci_tensor_free(output);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_reduce_max_all_keep_dim){
    unsigned int shape[] = {2, 3};
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor_alloc(first);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++)
        ((float *)first->data)[i] = static_cast<float>(i);
    adci_tensor *dim = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc(dim);
    ((int32_t *)dim->data)[0] = 0;
    ((int32_t *)dim->data)[1] = 1;
    adci_tensor *keep_dim = adci_tensor_init_vargs(1, ADCI_I32, 1);
    adci_tensor_alloc(keep_dim);
    adci_tensor_set_i32(keep_dim, 1, 0);
    adci_tensor *output = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_vector inputs = adci_vector_init(sizeof(adci_tensor*));
    adci_vector_add(&inputs, &first);
    adci_vector_add(&inputs, &dim);
    adci_vector_add(&inputs, &keep_dim);
    adci_tensor_reduce_max(inputs, output);
    EXPECT_EQ(output->n_dimension, first->n_dimension);    
    EXPECT_EQ(output->shape[0], 1);    
    EXPECT_EQ(output->shape[1], 1);    
    EXPECT_FLOAT_EQ(((float *)output->data)[0], static_cast<float>(5));
    adci_vector_free(&inputs);
    adci_tensor_free(keep_dim);
    adci_tensor_free(dim);
    adci_tensor_free(first);
    adci_tensor_free(output);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_reduce_max_all_dims){
    float tensor_values[] = {-4553.1489, -3505.6177, 26013.2051, 1696.8086, -5700.1772, -10731.9824, -6522.3354, -354.0594, -5777.6431, -2628.9851};
    unsigned int shape[] = {1, 10};
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor_alloc_set(first, tensor_values);
    adci_tensor *dim = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc(dim);
    ((int32_t *)dim->data)[0] = 0;
    ((int32_t *)dim->data)[1] = 1;
    adci_tensor *output = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_vector inputs = adci_vector_init(sizeof(adci_tensor*));
    adci_vector_add(&inputs, &first);
    adci_vector_add(&inputs, &dim);
    adci_tensor_reduce_max(inputs, output);
    EXPECT_EQ(output->n_dimension, 1);
    EXPECT_EQ(output->shape[0], 1);    
    EXPECT_FLOAT_EQ(((float *)output->data)[0], 26013.2051f);
    adci_vector_free(&inputs);
    adci_tensor_free(dim);
    adci_tensor_free(first);
    adci_tensor_free(output);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_concat){
    unsigned int shape[] = {2, 3};
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor_alloc(first);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++)
        ((float *)first->data)[i] = static_cast<float>(i);
    adci_tensor *second = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1] + 5);
    adci_tensor_alloc(second);
    for(unsigned int i = 0; i < shape[0] * (shape[1] + 5); i++)
        ((float *)second->data)[i] = static_cast<float>(i);
    adci_tensor *axis = adci_tensor_init_vargs(1, ADCI_I32, 1);
    adci_tensor_alloc(axis);
    adci_tensor_set_i32(axis, 1, 0);
    adci_tensor *output = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_vector inputs = adci_vector_init(sizeof(adci_tensor*));
    adci_vector_add(&inputs, &first); 
    adci_vector_add(&inputs, &second); 
    adci_vector_add(&inputs, &axis);
    adci_tensor_concat(inputs, output);
    EXPECT_EQ(output->n_dimension, 2);
    EXPECT_EQ(output->shape[0], shape[0]);
    EXPECT_EQ(output->shape[1], shape[1] + shape[1] + 5);
    for(unsigned int i = 0; i < output->shape[0] * output->shape[1]; i++){
        const unsigned int offset = i % output->shape[1];
        const unsigned int dim = i / output->shape[1];
        const float output_value = ((float*)output->data)[i];
        if(offset < shape[1]){
            const unsigned int index = dim * shape[1] + offset;
            EXPECT_FLOAT_EQ(output_value, ((float *)first->data)[index]);
        }else{
            const unsigned int index = dim * second->shape[1] + offset - shape[1];
            EXPECT_FLOAT_EQ(output_value, ((float *)second->data)[index]);
        }
    }
    adci_vector_free(&inputs);
    adci_tensor_free(first);
    adci_tensor_free(second);
    adci_tensor_free(axis);
    adci_tensor_free(output);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_mult_1d){
    unsigned int shape[] = {2, 3};
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor *mult = adci_tensor_init_vargs(1, ADCI_F32, shape[1]);
    adci_tensor_alloc(first);
    adci_tensor_alloc(mult);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++)
        ((float *)first->data)[i] = static_cast<float>(i);
    for(unsigned int i = 0; i < shape[1]; i++)
        adci_tensor_set_f32(mult, static_cast<float>(i), i);
    adci_tensor output;
    memset(&output, 0, sizeof(adci_tensor));
    adci_vector intputs = adci_vector_init(sizeof(adci_tensor *));
    adci_vector_add(&intputs, &first);
    adci_vector_add(&intputs, &mult);
    adci_tensor_mul(intputs, &output);
    for(unsigned int i = 0; i < shape[1]; i++)
        EXPECT_FLOAT_EQ(((float *)mult->data)[i], static_cast<float>(i));
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++)
        EXPECT_FLOAT_EQ(((float *)output.data)[i], ((float *)first->data)[i] * ((float *)mult->data)[i % shape[1]]);
    adci_vector_free(&intputs);
    adci_tensor_free(first);
    adci_tensor_free(mult);
    ADCI_FREE(output.data);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_mult_2d){
    unsigned int shape[] = {10, 2, 3};
    adci_tensor *first = adci_tensor_init(3, shape, ADCI_F32);
    adci_tensor *mult = adci_tensor_init_vargs(2, ADCI_F32, shape[1], shape[2]);
    adci_tensor_alloc(first);
    adci_tensor_alloc(mult);
    for(unsigned int i = 0; i < shape[0] * shape[1] * shape[2]; i++)
        ((float *)first->data)[i] = static_cast<float>(i);
    for(unsigned int i = 0; i < shape[1] * shape[2]; i++)
        ((float *)mult->data)[i] = static_cast<float>(i);
    adci_tensor output;
    memset(&output, 0, sizeof(adci_tensor));
    adci_vector intputs = adci_vector_init(sizeof(adci_tensor *));
    adci_vector_add(&intputs, &first);
    adci_vector_add(&intputs, &mult);
    adci_tensor_mul(intputs, &output);
    for(unsigned int i = 0; i < shape[1] * shape[2]; i++)
        EXPECT_FLOAT_EQ(((float *)mult->data)[i], static_cast<float>(i));
    for(unsigned int i = 0; i < shape[0] * shape[1] * shape[2]; i++)
        EXPECT_FLOAT_EQ(((float *)output.data)[i], ((float *)first->data)[i] * ((float *)mult->data)[i % (shape[1] * shape[2])]);
    adci_vector_free(&intputs);
    adci_tensor_free(first);
    adci_tensor_free(mult);
    ADCI_FREE(output.data);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_max_pool2D){
    unsigned int shape[] = {1, 1, 4, 4};
    float values[][4] = {
        {1.f, 2.f , 5.f , 6.f },
        {0.f, 6.f , 10.f, 6.f },
        {1.f, 20.f, 5.f , 6.f },
        {1.f, 2.f , 5.f , 30.f},
    };
    adci_tensor *tensor = adci_tensor_init(sizeof(shape)/sizeof(unsigned int), shape, ADCI_F32);
    adci_tensor_alloc_set(tensor, values);
    unsigned int size_data[] = {2, 2};
    adci_tensor *size   = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc_set(size, size_data);
    unsigned int stride_data[] = {2, 2};
    adci_tensor *stride = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc_set(stride, stride_data);
    unsigned int dims_data[] = {2, 3};
    adci_tensor *dims = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc_set(dims, dims_data);
    adci_tensor output;
    memset(&output, 0, sizeof(adci_tensor));

    adci_vector inputs = adci_vector_init(sizeof(adci_tensor *));
    adci_vector_add(&inputs, &tensor);
    adci_vector_add(&inputs, &size);
    adci_vector_add(&inputs, &stride);
    adci_vector_add(&inputs, &dims);
    adci_tensor_max_pool2D(inputs, &output);
    EXPECT_EQ(output.n_dimension, 4);
    EXPECT_EQ(output.shape[0], 1);
    EXPECT_EQ(output.shape[1], 1);
    EXPECT_EQ(output.shape[2], 2);
    EXPECT_EQ(output.shape[3], 2);
    EXPECT_EQ(output.dtype, tensor->dtype);
    EXPECT_FLOAT_EQ(((float *)output.data)[0], 6.f);
    EXPECT_FLOAT_EQ(((float *)output.data)[1], 10.f);
    EXPECT_FLOAT_EQ(((float *)output.data)[2], 20.f);
    EXPECT_FLOAT_EQ(((float *)output.data)[3], 30.f);
    adci_vector_free(&inputs);
    adci_tensor_free(tensor);
    adci_tensor_free(size);
    adci_tensor_free(stride);
    adci_tensor_free(dims);
    ADCI_FREE(output.data);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_max_pool2D_BHWC){
    unsigned int shape[] = {1, 4, 4, 1};
    float values[][4] = {
        {1.f, 2.f , 5.f , 6.f },
        {0.f, 6.f , 10.f, 6.f },
        {1.f, 20.f, 5.f , 6.f },
        {1.f, 2.f , 5.f , 30.f},
    };
    adci_tensor *tensor = adci_tensor_init(sizeof(shape)/sizeof(unsigned int), shape, ADCI_F32);
    adci_tensor_alloc_set(tensor, values);
    unsigned int size_data[] = {2, 2};
    adci_tensor *size   = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc_set(size, size_data);
    unsigned int stride_data[] = {2, 2};
    adci_tensor *stride = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc_set(stride, stride_data);
    unsigned int dims_data[] = {1, 2};
    adci_tensor *dims = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc_set(dims, dims_data);
    adci_tensor output;
    memset(&output, 0, sizeof(adci_tensor));

    adci_vector inputs = adci_vector_init(sizeof(adci_tensor *));
    adci_vector_add(&inputs, &tensor);
    adci_vector_add(&inputs, &size);
    adci_vector_add(&inputs, &stride);
    adci_vector_add(&inputs, &dims);
    adci_tensor_max_pool2D(inputs, &output);
    EXPECT_EQ(output.n_dimension, 4);
    EXPECT_EQ(output.shape[0], 1);
    EXPECT_EQ(output.shape[1], 2);
    EXPECT_EQ(output.shape[2], 2);
    EXPECT_EQ(output.shape[3], 1);
    EXPECT_EQ(output.dtype, tensor->dtype);
    EXPECT_FLOAT_EQ(((float *)output.data)[0], 6.f);
    EXPECT_FLOAT_EQ(((float *)output.data)[1], 10.f);
    EXPECT_FLOAT_EQ(((float *)output.data)[2], 20.f);
    EXPECT_FLOAT_EQ(((float *)output.data)[3], 30.f);
    adci_vector_free(&inputs);
    adci_tensor_free(tensor);
    adci_tensor_free(size);
    adci_tensor_free(stride);
    adci_tensor_free(dims);
    ADCI_FREE(output.data);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_avg_pool2D){
    unsigned int shape[] = {1, 1, 4, 4};
    float values[][4] = {
        {1.f, 2.f , 5.f , 6.f },
        {0.f, 6.f , 10.f, 6.f },
        {1.f, 20.f, 5.f , 6.f },
        {1.f, 2.f , 5.f , 30.f},
    };
    adci_tensor *tensor = adci_tensor_init(sizeof(shape)/sizeof(unsigned int), shape, ADCI_F32);
    adci_tensor_alloc_set(tensor, values);
    unsigned int size_data[] = {2, 2};
    adci_tensor *size   = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc_set(size, size_data);
    unsigned int stride_data[] = {2, 2};
    adci_tensor *stride = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc_set(stride, stride_data);
    unsigned int dims_data[] = {2, 3};
    adci_tensor *dims = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc_set(dims, dims_data);
    adci_tensor output;
    memset(&output, 0, sizeof(adci_tensor));
    adci_tensor_avg_pool2D_args(tensor, size, stride, dims, &output);
    EXPECT_EQ(output.n_dimension, 4);
    EXPECT_EQ(output.shape[0], 1);
    EXPECT_EQ(output.shape[1], 1);
    EXPECT_EQ(output.shape[2], 2);
    EXPECT_EQ(output.shape[3], 2);
    EXPECT_EQ(output.dtype, tensor->dtype);
    EXPECT_FLOAT_EQ(((float *)output.data)[0], 2.25f);
    EXPECT_FLOAT_EQ(((float *)output.data)[1], 6.75f);
    EXPECT_FLOAT_EQ(((float *)output.data)[2], 6.0f);
    EXPECT_FLOAT_EQ(((float *)output.data)[3], 11.5f);
    adci_tensor_free(tensor);
    adci_tensor_free(size);
    adci_tensor_free(stride);
    adci_tensor_free(dims);
    ADCI_FREE(output.data);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_conv2D_BWHC){
    unsigned int shape[] = {1 ,4, 4, 1};
    float values[][4] = {
        {1.f, 2.f , 5.f , 6.f },
        {0.f, 6.f , 10.f, 6.f },
        {1.f, 20.f, 5.f , 6.f },
        {1.f, 2.f , 5.f , 30.f},
    };
    adci_tensor *tensor = adci_tensor_init(4, shape, ADCI_F32);
    adci_tensor_alloc_set(tensor, values);
    unsigned int filter_shape[] = {1, 2, 2, 1};
    adci_tensor *filter   = adci_tensor_init(4, filter_shape, ADCI_F32);
    adci_tensor_alloc(filter);
    for(unsigned int i = 0; i < 4; i++) ((float *)filter->data)[i] = 1.0f;
    unsigned int stride_data[] = {2, 2};
    adci_tensor *stride = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc_set(stride, stride_data);
    unsigned int dims_data[] = {1, 2, 3};
    adci_tensor *dims = adci_tensor_init_vargs(1, ADCI_I32, 3);
    adci_tensor_alloc_set(dims, dims_data);
    adci_tensor output;
    memset(&output, 0, sizeof(adci_tensor));

    adci_vector inputs = adci_vector_init(sizeof(adci_tensor *));
    adci_vector_add(&inputs, &tensor);
    adci_vector_add(&inputs, &filter);
    adci_vector_add(&inputs, &stride);
    adci_vector_add(&inputs, &dims);

    adci_tensor_conv2D(inputs, &output);
    EXPECT_EQ(output.n_dimension, 4);
    EXPECT_EQ(output.shape[0], 1);
    EXPECT_EQ(output.shape[1], 2);
    EXPECT_EQ(output.shape[2], 2);
    EXPECT_EQ(output.shape[3], 1);
    EXPECT_EQ(output.dtype, tensor->dtype);
    EXPECT_FLOAT_EQ(((float *)output.data)[0], 9.f);
    EXPECT_FLOAT_EQ(((float *)output.data)[1], 27.f);
    EXPECT_FLOAT_EQ(((float *)output.data)[2], 24.f);
    EXPECT_FLOAT_EQ(((float *)output.data)[3], 46.f);
    adci_vector_free(&inputs);
    adci_tensor_free(tensor);
    adci_tensor_free(filter);
    adci_tensor_free(stride);
    adci_tensor_free(dims);
    ADCI_FREE(output.data);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_conv2D_BCWH){
    unsigned int shape[] = {1, 1, 4, 4};
    float values[][4] = {
        {1.f, 2.f , 5.f , 6.f },
        {0.f, 6.f , 10.f, 6.f },
        {1.f, 20.f, 5.f , 6.f },
        {1.f, 2.f , 5.f , 30.f},
    };
    adci_tensor *tensor = adci_tensor_init(4, shape, ADCI_F32);
    adci_tensor_alloc_set(tensor, values);
    /* FILTER SHAPE DOES NOT CHANGE DEPENDING ON THE INPUT SHAPE, ALWAYS [O_CHANNEL, WIDTH, HEIGHT, I_CHANNEL] */
    unsigned int filter_shape[] = {1, 2, 2, 1};
    adci_tensor *filter   = adci_tensor_init(4, filter_shape, ADCI_F32);
    adci_tensor_alloc(filter);
    for(unsigned int i = 0; i < 4; i++) ((float *)filter->data)[i] = 1.0f;
    unsigned int stride_data[] = {2, 2};
    adci_tensor *stride = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc_set(stride, stride_data);
    unsigned int dims_data[] = {2, 3, 1};
    adci_tensor *dims = adci_tensor_init_vargs(1, ADCI_I32, 3);
    adci_tensor_alloc_set(dims, dims_data);
    adci_tensor output;
    memset(&output, 0, sizeof(adci_tensor));

    adci_vector inputs = adci_vector_init(sizeof(adci_tensor *));
    adci_vector_add(&inputs, &tensor);
    adci_vector_add(&inputs, &filter);
    adci_vector_add(&inputs, &stride);
    adci_vector_add(&inputs, &dims);

    adci_tensor_conv2D(inputs, &output);
    EXPECT_EQ(output.n_dimension, 4);
    EXPECT_EQ(output.shape[0], 1);
    EXPECT_EQ(output.shape[1], 1);
    EXPECT_EQ(output.shape[2], 2);
    EXPECT_EQ(output.shape[3], 2);
    EXPECT_EQ(output.dtype, tensor->dtype);
    EXPECT_FLOAT_EQ(((float *)output.data)[0], 9.f);
    EXPECT_FLOAT_EQ(((float *)output.data)[1], 27.f);
    EXPECT_FLOAT_EQ(((float *)output.data)[2], 24.f);
    EXPECT_FLOAT_EQ(((float *)output.data)[3], 46.f);
    adci_vector_free(&inputs);
    adci_tensor_free(tensor);
    adci_tensor_free(filter);
    adci_tensor_free(stride);
    adci_tensor_free(dims);
    ADCI_FREE(output.data);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_transpose){
    unsigned int shape[] = {1 ,2, 3, 1};
    float values[][3] = {
        {1.f, 2.f , 5.f },
        {0.f, 6.f , 10.f},
    };
    float expected_values[][2] = {
        {1.f, 0.f},
        {2.f, 6.f},
        {5.f, 10.f}
    };
    adci_tensor *tensor = adci_tensor_init(4, shape, ADCI_F32);
    adci_tensor_alloc_set(tensor, values);
    adci_tensor *dims = adci_tensor_init_vargs(1, ADCI_I32, 4);
    unsigned int dims_data[] = {0, 2, 1, 3};
    adci_tensor_alloc_set(dims, dims_data);
    adci_tensor output;
    memset(&output, 0, sizeof(adci_tensor));
    adci_vector inputs = adci_vector_init(sizeof(adci_tensor *));
    adci_vector_add(&inputs, &tensor);
    adci_vector_add(&inputs, &dims);

    adci_tensor_transpose(inputs, &output);
    EXPECT_EQ(output.n_dimension, 4);
    EXPECT_EQ(output.shape[0], 1);
    EXPECT_EQ(output.shape[1], 3);
    EXPECT_EQ(output.shape[2], 2);
    EXPECT_EQ(output.shape[3], 1);
    EXPECT_EQ(output.dtype, tensor->dtype);
    for(unsigned int i = 0; i < 6; i++)
        EXPECT_FLOAT_EQ(((float *)output.data)[i], ((float *)expected_values)[i]);
    adci_vector_free(&inputs);
    adci_tensor_free(tensor);
    adci_tensor_free(dims);
    ADCI_FREE(output.data);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_fully_connected){
    float values[][4] = {
        {1.f, 2.f , 5.f , 6.f},
        {0.f, 6.f , 10.f, 3.f},
    };
    float expected[] = { 14.f, 19.f };
    adci_tensor *tensor = adci_tensor_init_vargs(2, ADCI_F32, 1, 4);
    adci_tensor_alloc(tensor);
    for(unsigned int i = 0; i < tensor->shape[1]; i++)
        ((float *)tensor->data)[i] = 1.f;
    adci_tensor *weights = adci_tensor_init_vargs(2, ADCI_F32, 2, 4);
    adci_tensor_alloc_set(weights, values);
    adci_tensor output;
    memset(&output, 0, sizeof(adci_tensor));

    adci_vector inputs = adci_vector_init(sizeof(adci_tensor *));
    adci_vector_add(&inputs, &tensor);
    adci_vector_add(&inputs, &weights);

    adci_tensor_fully_connected(inputs, &output);
    EXPECT_EQ(output.n_dimension, 2);
    EXPECT_EQ(output.shape[0], 1);
    EXPECT_EQ(output.shape[1], 2);
    EXPECT_EQ(output.dtype, tensor->dtype);
    EXPECT_FLOAT_EQ(((float *)output.data)[0], expected[0]);
    EXPECT_FLOAT_EQ(((float *)output.data)[1], expected[1]);
    adci_vector_free(&inputs);
    adci_tensor_free(tensor);
    adci_tensor_free(weights);
    ADCI_FREE(output.data);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_conv2d_tflite_comp){
    const float input_data[][5][5][1] = {{
        {{2}, {1}, {2}, {0}, {1}},
        {{1}, {3}, {2}, {2}, {3}},
        {{1}, {1}, {3}, {3}, {0}},
        {{2}, {2}, {0}, {1}, {1}},
        {{0}, {0}, {3}, {1}, {2}}, }};
    const float filter_data[][2][1][2] = {
        { {{2, 0.1}}, {{3, 0.2}} },
        { {{0, 0.3}}, {{1, 0.4}} }, };
    const float expected_data[][4][4][2] = {{
       {{10.f, 1.9f},
        {10.f, 2.2f},
        { 6.f, 1.6f},
        { 6.f, 2.0f}},
       {{12.f, 1.4f},
        {15.f, 2.2f},
        {13.f, 2.7f},
        {13.f, 1.7f}},
       {{ 7.f, 1.7f},
        {11.f, 1.3f},
        {16.f, 1.3f},
        { 7.f, 1.0f}},
       {{10.f, 0.6f},
        { 7.f, 1.4f},
        { 4.f, 1.5f},
        { 7.f, 1.40f}}
    }};
    adci_tensor *tensor = adci_tensor_init_vargs(4, ADCI_F32, 1, 5, 5, 1);
    adci_tensor_alloc_set(tensor, input_data);
    adci_tensor *filter = adci_tensor_init_vargs(4, ADCI_F32, 2, 2, 1, 2);
    adci_tensor_alloc_set(filter, filter_data);
    
    /* PUT THE FILTER IN THE RIGHT FORMAT */
    unsigned int transpose_data[] = {3, 1, 0, 2};
    adci_tensor *transpose_dims = adci_tensor_init_vargs(1, ADCI_I32, 4);
    adci_tensor_alloc_set(transpose_dims, transpose_data);
    adci_tensor_transpose_args(filter, transpose_dims, filter);

    adci_tensor *stride = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc(stride);
    adci_tensor_set_i32(stride, 1, 0);
    adci_tensor_set_i32(stride, 1, 1);

    adci_tensor *dims = adci_tensor_init_vargs(1, ADCI_I32, 3);
    adci_tensor_alloc(dims);
    adci_tensor_set_i32(dims, 2, 0);
    adci_tensor_set_i32(dims, 1, 1);
    adci_tensor_set_i32(dims, 3, 2);

    adci_tensor output;
    memset(&output, 0, sizeof(adci_tensor));

    adci_tensor_conv2D_args(tensor, filter, stride, dims, &output);
    EXPECT_EQ(output.n_dimension, 4);
    EXPECT_EQ(output.shape[0], 1);
    EXPECT_EQ(output.shape[1], 4);
    EXPECT_EQ(output.shape[2], 4);
    EXPECT_EQ(output.shape[3], 2);

    for(unsigned int i = 0; i < 32; i++)
        EXPECT_FLOAT_EQ(((float*)expected_data)[i], ((float *)output.data)[i]);

    ADCI_FREE(output.data);
    adci_tensor_free(dims);
    adci_tensor_free(transpose_dims);
    adci_tensor_free(stride);
    adci_tensor_free(filter);
    adci_tensor_free(tensor);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_batch_mathmult_same_size){
    adci_tensor *first = adci_tensor_init_vargs(3, ADCI_F32, 1, 4, 4);
    adci_tensor_alloc(first);
    const float first_value = 1.f;
    adci_tensor_fill(first, &first_value);
    adci_tensor *second = adci_tensor_init_vargs(3, ADCI_F32, 1, 4, 4);
    adci_tensor_alloc(second);
    for(unsigned int i = 0; i < second->shape[1] * second->shape[2]; i++)
        ((float*)second->data)[i] = static_cast<float>(i);
    adci_tensor output;
    memset(&output, 0, sizeof(adci_tensor));
    adci_tensor_batch_matmult_args(first, second, &output);
    EXPECT_EQ(output.shape[0], 1);
    EXPECT_EQ(output.shape[1], first->shape[1]);
    EXPECT_EQ(output.shape[2], second->shape[2]);
    float expected_row[] = {24.0000f, 28.0000f, 32.0000f, 36.0000f};
    for(unsigned int i = 0; i < output.shape[1] * output.shape[2]; i++)
        EXPECT_FLOAT_EQ(((float *)output.data)[i], expected_row[i % (sizeof(expected_row) / sizeof(float))]);
    ADCI_FREE(output.data);
    adci_tensor_free(first);
    adci_tensor_free(second);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_batch_mathmult){
    adci_tensor *first = adci_tensor_init_vargs(3, ADCI_F32, 1, 2, 6);
    float first_values[1][2][6] = {{
        {2.f, 6.f, 8.f, 1.2f, 7.5f, 4.6f},
        {8.9f, 4.5f, 2.7, 5.0f, 8.99f, 0.46f}
    }};
    adci_tensor_alloc_set(first, first_values);
    adci_tensor *second = adci_tensor_init_vargs(3, ADCI_F32, 1, 6, 3);
    float second_values[1][6][3] = {{
        { 2.f, 3.0f, 1.0f},
        {2.5f, 6.7f, 9.5f},
        {4.6f, 2.5f, 4.3f},
        {7.9f, 6.7f, 3.5f},
        {0.6f, 4.6f, 2.4f},
        {9.5f, 5.6f, 4.6f}
    }};
    adci_tensor_alloc_set(second, second_values);
    adci_tensor output;
    memset(&output, 0, sizeof(adci_tensor));
    adci_tensor_batch_matmult_args(first, second, &output);
    EXPECT_EQ(output.shape[0], 1);
    EXPECT_EQ(output.shape[1], first->shape[1]);
    EXPECT_EQ(output.shape[2], second->shape[2]);
    float expected_output[1][2][3] = {{
        {113.479996f, 134.5f, 136.76f  },
        {90.734f, 141.03f, 104.451996f}
    }};
    for(unsigned int i = 0; i < output.shape[1] * output.shape[2]; i++)
        EXPECT_EQ(((float*)output.data)[i], ((float*)expected_output)[i]);
    ADCI_FREE(output.data);
    adci_tensor_free(first);
    adci_tensor_free(second);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_argmax){
    adci_tensor *tensor = adci_tensor_init_vargs(3, ADCI_F32, 1, 6, 3);
    float values[1][6][3] = {{
        { 2.f, 3.0f, 1.0f},
        {2.5f, 6.7f, 9.5f},
        {4.6f, 2.5f, 4.3f},
        {7.9f, 6.9f, 3.5f},
        {0.6f, 4.6f, 2.4f},
        {9.5f, 5.6f, 4.6f}
    }};
    adci_tensor_alloc_set(tensor, values);
    adci_tensor *dim = adci_tensor_init_vargs(1, ADCI_I32, 1);
    adci_tensor_alloc(dim);
    adci_tensor_set_i32(dim, 1, 0);
    adci_tensor output;
    memset(&output, 0, sizeof(adci_tensor));
    adci_tensor_argmax_args(tensor, dim, NULL, &output);
    EXPECT_EQ(output.n_dimension, 2);
    EXPECT_EQ(output.shape[0], 1);
    EXPECT_EQ(output.shape[1], 3);
    uint32_t expected_output[1][3] = {{ 5, 3, 1 }};
    for(unsigned int i = 0; i < 3; i++)
        EXPECT_EQ(((uint32_t *)output.data)[i], ((uint32_t *)expected_output)[i]);
    ADCI_FREE(output.data);
    adci_tensor_free(tensor);
    adci_tensor_free(dim);
}

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_compute_op){
    /* TODO, IMPLEMENT */
}