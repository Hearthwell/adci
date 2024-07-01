#ifndef ADCI_TENSOR_COMMON_H
#define ADCI_TENSOR_COMMON_H

/* COMMON PRIVATE ATI_TENSOR FUNCTIONS */

#include "adci_tensor.h"

unsigned int adci_tensor_element_count_ext(unsigned int n_dims, const unsigned int *shape);

struct adci_multi_dim_counter{
    const struct adci_tensor *tensor;
    unsigned int free_dims_count;
    unsigned int precomputed_volumes[ADCI_TENSOR_MAX_DIM];
    unsigned int dim_indeces[ADCI_TENSOR_MAX_DIM];
    unsigned int counter[ADCI_TENSOR_MAX_DIM];
};

struct adci_multi_dim_counter adci_tensor_init_multidim_counter(const struct adci_tensor *tensor, const unsigned int *dims, unsigned int length);
void adci_tensor_increase_counter(struct adci_multi_dim_counter *counter);
unsigned int adci_tensor_get_counter_offset(struct adci_multi_dim_counter counter);

#endif //ADCI_TENSOR_COMMON_H