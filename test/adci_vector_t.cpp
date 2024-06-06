#include <gtest/gtest.h>

extern "C"{
#include "adci_common.h"
}

#define ADCI_VECTOR_SUITE_NAME ADCI_VECTOR

TEST(ADCI_VECTOR_SUITE_NAME, adci_vector_init){
    struct adci_vector * vector = adci_vector_init(sizeof(unsigned int));
    EXPECT_NE(vector->data, nullptr);
    EXPECT_EQ(vector->length, 0);
    EXPECT_EQ(vector->capacity, 10);
    EXPECT_EQ(vector->bsize, sizeof(unsigned int));
    adci_vector_free(vector);
}

TEST(ADCI_VECTOR_SUITE_NAME, adci_vector_add){
    struct adci_vector * vector = adci_vector_init(sizeof(unsigned int));
    const unsigned int value = 107;
    adci_vector_add(vector, &value);
    EXPECT_EQ(*((unsigned int *)vector->data), value);
    EXPECT_EQ(vector->length, 1);
    adci_vector_free(vector);
}

TEST(ADCI_VECTOR_SUITE_NAME, adci_vector_get){
    struct adci_vector * vector = adci_vector_init(sizeof(unsigned int));
    const unsigned int value = 109;
    adci_vector_add(vector, &value);
    EXPECT_EQ(*((unsigned int *)vector->data), value);
    EXPECT_EQ(vector->length, 1);
    unsigned int *element = (unsigned int *)adci_vector_get(vector, 0);
    EXPECT_EQ(*element, value);
    adci_vector_free(vector);
}

TEST(ADCI_VECTOR_SUITE_NAME, adci_vector_remove){
    struct adci_vector * vector = adci_vector_init(sizeof(unsigned int));
    const unsigned int value = 109;
    adci_vector_add(vector, &value);
    EXPECT_EQ(*((unsigned int *)vector->data), value);
    EXPECT_EQ(vector->length, 1);
    bool status = adci_vector_remove(vector, &value);
    EXPECT_TRUE(status);
    EXPECT_EQ(vector->length, 0);
    adci_vector_free(vector);
}