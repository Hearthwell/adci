#include <gtest/gtest.h>

extern "C"{
#include "adci_tensor_common.h"
}

#define ADCI_TENSOR_COMMON_SUITE_NAME ADCI_TENSOR_COMMON

TEST(ADCI_TENSOR_COMMON_SUITE_NAME, adci_tensor_init_multidim_counter){
    unsigned int shape[] = {1, 10, 15};
    adci_tensor *tensor = adci_tensor_init(sizeof(shape) / sizeof(unsigned int), shape, ADCI_I32);

    unsigned int dims[] = {1};
    struct adci_multi_dim_counter couter = adci_tensor_init_multidim_counter(tensor, dims, 1);
    EXPECT_EQ(couter.tensor, tensor);
    EXPECT_EQ(couter.free_dims_count, 1);
    for(unsigned int i = 0; i < couter.free_dims_count; i++)
        EXPECT_EQ(couter.counter[i], 0);
    EXPECT_EQ(couter.dim_indeces[0], 1);
    EXPECT_EQ(couter.precomputed_volumes[0], 150);
    EXPECT_EQ(couter.precomputed_volumes[1], 15);
    EXPECT_EQ(couter.precomputed_volumes[2], 1);
    adci_tensor_free(tensor);
}

TEST(ADCI_TENSOR_COMMON_SUITE_NAME, adci_tensor_get_counter_offset){
    unsigned int shape[] = {2, 10, 15};
    adci_tensor *tensor = adci_tensor_init(sizeof(shape) / sizeof(unsigned int), shape, ADCI_I32);
    unsigned int dims[] = {0, 1};
    struct adci_multi_dim_counter counter = adci_tensor_init_multidim_counter(tensor, dims, 2);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++){
        const unsigned int expected_offset = (i / shape[1]) * shape[1] * shape[2] + (i % shape[1]) * shape[2];
        const unsigned int offset = adci_tensor_get_counter_offset(counter);
        adci_tensor_increase_counter(&counter);
        EXPECT_EQ(expected_offset, offset);
    }
    adci_tensor_free(tensor);
}

