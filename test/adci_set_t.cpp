#include <gtest/gtest.h>

extern "C"{
#include "adci_common.h"
}

#define ADCI_SET_SUITE_NAME ADCI_SET_SUITE_NAME

TEST(ADCI_SET_SUITE_NAME, adci_set_init){
    adci_set set = adci_set_init(sizeof(unsigned int *), NULL);
    EXPECT_EQ(set.bsize, sizeof(unsigned int *));
    EXPECT_NE(set.capacity, 0);
    EXPECT_NE(set.data, nullptr);
    EXPECT_NE(set.hasher, nullptr);
    EXPECT_EQ(set.length, 0);
    adci_set_free(&set);
}

TEST(ADCI_SET_SUITE_NAME, adci_set_add){
    adci_set set = adci_set_init(sizeof(unsigned int *), NULL);
    const unsigned int count = 10;
    for(unsigned int i = 0; i < count; i++){
        uint64_t value = i;
        adci_set_add(&set, &value);   
    }
    EXPECT_EQ(set.length, count);
    adci_set_free(&set);
}

TEST(ADCI_SET_SUITE_NAME, adci_set_add_large){
    adci_set set = adci_set_init(sizeof(unsigned int *), NULL);
    const unsigned int count = 100;
    for(unsigned int i = 0; i < count; i++){
        uint64_t value = i;
        adci_set_add(&set, &value);   
    }
    EXPECT_EQ(set.length, count);
    adci_set_free(&set);
}

TEST(ADCI_SET_SUITE_NAME, adci_set_has){
    adci_set set = adci_set_init(sizeof(unsigned int *), NULL);
    const unsigned int count = 10;
    for(unsigned int i = 0; i < count; i++){
        uint64_t value = i;
        adci_set_add(&set, &value);   
    }
    for(unsigned int i = 0; i < count; i++){
        uint64_t value = i;
        EXPECT_TRUE(adci_set_has(set, &value));   
    }
    uint64_t value = count + 1;
    EXPECT_FALSE(adci_set_has(set, (unsigned int *)&value)); 
    adci_set_free(&set);
}

TEST(ADCI_SET_SUITE_NAME, adci_set_iterator){
    adci_set set = adci_set_init(sizeof(unsigned int *), NULL);
    const unsigned int count = 10;
    for(unsigned int i = 0; i < count; i++){
        uint64_t value = i;
        adci_set_add(&set, &value);   
    }
    struct adci_set_iterator iter = adci_set_get_iterator(&set);
    unsigned int iteration_count = 0;
    do{
        adci_set_get_next(&iter);
        iteration_count++;
    }while(!iter.done);
    EXPECT_EQ(iteration_count - 1, count);
    adci_set_free(&set);
}