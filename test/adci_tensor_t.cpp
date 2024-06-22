#include <gtest/gtest.h>

extern "C"{
#include "adci_tensor.h"   
}

#define TEST_SUITE_NAME ADCI_TENSOR

TEST(TEST_SUITE_NAME, adci_tensor_print){
    adci_tensor *tensor = adci_tensor_init_2d(6, 5, ADCI_F32);
    adci_tensor_alloc(tensor);
    for(unsigned int i = 0; i < 6 * 5; i++) ((float *)tensor->data)[i] = static_cast<float>(i);
    adci_tensor_print(tensor);
    adci_tensor_free(tensor);
    /* TODO ADD TESTS TO OUTPUT OF PRINT */
}

TEST(TEST_SUITE_NAME, adci_tensor_set_element_i32){
    int32_t value = 10;
    adci_tensor *tensor = adci_tensor_init_2d(5, 6, ADCI_I32);
    adci_tensor_alloc(tensor);
    for(unsigned int i = 0; i < 5 * 6; i++)
        ((int32_t*)tensor->data)[i] = 0;
    adci_tensor_set_i32(tensor, value, 0, 0);
    EXPECT_EQ(((int32_t*)tensor->data)[0], value);
    adci_tensor_free(tensor);
}

TEST(TEST_SUITE_NAME, adci_tensor_set_element_f32){
    float value = 98.45;
    adci_tensor *tensor = adci_tensor_init_2d(5, 6, ADCI_F32);
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
    adci_tensor *tensor = adci_tensor_init_2d(5, 6, ADCI_F32);
    adci_tensor_alloc(tensor);
    for(unsigned int i = 0; i < 5 * 6; i++)
        ((float*)tensor->data)[i] = 0.f;
    adci_tensor_set_element(tensor, &value, 3, 1);
    EXPECT_FLOAT_EQ(((float*)tensor->data)[0], 0.f);
    EXPECT_FLOAT_EQ(((float*)tensor->data)[3 * 6 + 1], value);
    adci_tensor_free(tensor);
}