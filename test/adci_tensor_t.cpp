#include <gtest/gtest.h>

extern "C"{
#include "adci_tensor.h"   
}

#define TEST_SUITE_NAME ADCI_TENSOR

TEST(TEST_SUITE_NAME, adci_tensor_init_vargs){
    adci_tensor *tensor = adci_tensor_init_vargs(4, ADCI_I32, 4, 3, 2, 1);
    EXPECT_EQ(tensor->n_dimension, 4);
    EXPECT_EQ(tensor->shape[0], 4);
    EXPECT_EQ(tensor->shape[1], 3);
    EXPECT_EQ(tensor->shape[2], 2);
    EXPECT_EQ(tensor->shape[3], 1);
    adci_tensor_free(tensor);
}

TEST(TEST_SUITE_NAME, adci_tensor_print){
    adci_tensor *tensor = adci_tensor_init_vargs(2, ADCI_F32, 6, 5);
    adci_tensor_alloc(tensor);
    for(unsigned int i = 0; i < 6 * 5; i++) ((float *)tensor->data)[i] = static_cast<float>(i);
    adci_tensor_print(tensor);
    adci_tensor_free(tensor);
    /* TODO ADD TESTS TO OUTPUT OF PRINT */
}

TEST(TEST_SUITE_NAME, adci_tensor_set_element_i32){
    int32_t value = 10;
    adci_tensor *tensor = adci_tensor_init_vargs(2, ADCI_I32, 5, 6);
    adci_tensor_alloc(tensor);
    for(unsigned int i = 0; i < 5 * 6; i++)
        ((int32_t*)tensor->data)[i] = 0;
    adci_tensor_set_i32(tensor, value, 0, 0);
    EXPECT_EQ(((int32_t*)tensor->data)[0], value);
    adci_tensor_free(tensor);
}

TEST(TEST_SUITE_NAME, adci_tensor_set_element_f32){
    float value = 98.45;
    adci_tensor *tensor = adci_tensor_init_vargs(2, ADCI_F32, 5, 6);
    adci_tensor_alloc(tensor);
    for(unsigned int i = 0; i < 5 * 6; i++)
        ((float*)tensor->data)[i] = 0.f;
    adci_tensor_set_f32(tensor, value, 3, 1);
    EXPECT_FLOAT_EQ(((float*)tensor->data)[0], 0.f);
    EXPECT_FLOAT_EQ(((float*)tensor->data)[3 * 6 + 1], value);
    adci_tensor_free(tensor);
}

TEST(TEST_SUITE_NAME, adci_tensor_set_element_generic){
    float value = 98.45;
    adci_tensor *tensor = adci_tensor_init_vargs(2, ADCI_F32, 5, 6);
    adci_tensor_alloc(tensor);
    for(unsigned int i = 0; i < 5 * 6; i++)
        ((float*)tensor->data)[i] = 0.f;
    adci_tensor_set_element(tensor, &value, 3, 1);
    EXPECT_FLOAT_EQ(((float*)tensor->data)[0], 0.f);
    EXPECT_FLOAT_EQ(((float*)tensor->data)[3 * 6 + 1], value);
    adci_tensor_free(tensor);
}

TEST(TEST_SUITE_NAME, adci_tensor_get_element_generic){
    adci_tensor *tensor = adci_tensor_init_vargs(2, ADCI_F32, 5, 6);
    adci_tensor_alloc(tensor);
    for(unsigned int i = 0; i < 5 * 6; i++)
        ((float*)tensor->data)[i] = (float)i;
    void *element = adci_tensor_get_element(tensor, 0, 5);
    EXPECT_FLOAT_EQ(((float*)element)[0], 5.f);
    adci_tensor_free(tensor);
}

TEST(TEST_SUITE_NAME, adci_tensor_get_f32){
    adci_tensor *tensor = adci_tensor_init_vargs(2, ADCI_F32, 5, 6);
    adci_tensor_alloc(tensor);
    for(unsigned int i = 0; i < 5 * 6; i++)
        ((float*)tensor->data)[i] = (float)i;
    float element = adci_tensor_get_f32(tensor, 1, 5);
    EXPECT_FLOAT_EQ(element, 11.f);
    adci_tensor_free(tensor);
}

TEST(TEST_SUITE_NAME, adci_tensor_get_i32){
    adci_tensor *tensor = adci_tensor_init_vargs(2, ADCI_I32, 5, 6);
    adci_tensor_alloc(tensor);
    for(unsigned int i = 0; i < 5 * 6; i++)
        ((int32_t *)tensor->data)[i] = i;
    int32_t element = adci_tensor_get_i32(tensor, 1, 5);
    EXPECT_EQ(element, 11);
    adci_tensor_free(tensor);
}

TEST(TEST_SUITE_NAME, adci_tensor_fill){
    adci_tensor *tensor = adci_tensor_init_vargs(2, ADCI_I32, 5, 6);
    adci_tensor_alloc(tensor);
    for(unsigned int i = 0; i < 5 * 6; i++)
        ((int32_t *)tensor->data)[i] = i;
    int32_t value = 0;
    adci_tensor_fill(tensor, &value);
    for(unsigned int i = 0; i < 5 * 6; i++)
        EXPECT_EQ(((int32_t *)tensor->data)[i], value);
    adci_tensor_free(tensor);
}