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
    adci_tensor *first = adci_tensor_init_2d(shape[0], shape[1], ADCI_F32);
    adci_tensor *shape_t = adci_tensor_init_1d(n_dims, ADCI_I32);
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
    adci_tensor *first = adci_tensor_init_2d(shape[0], shape[1], ADCI_F32);
    adci_tensor_alloc(first);
    adci_tensor *shape_t = adci_tensor_init_1d(n_dims, ADCI_I32);
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
    adci_tensor *first = adci_tensor_init_2d(shape[0], shape[1], ADCI_F32);
    adci_tensor_alloc(first);
    adci_tensor *second = adci_tensor_init_2d(shape[0], shape[1], ADCI_F32);
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

TEST(ADCI_TENSOR_OP_SUITE_NAME, adci_tensor_compute_op){
    /* TODO, IMPLEMENT */
}