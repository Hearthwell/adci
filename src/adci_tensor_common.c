#include "adci_tensor_common.h"

unsigned int adci_tensor_element_count_ext(unsigned int n_dims, const unsigned int *shape){
    unsigned int size = 1;
    for(unsigned int i = 0; i < n_dims; i++)
        size *= shape[i];
    return size;
}