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