#include "adci_tensor_common.h"
#include "adci_common.h"

unsigned int adci_tensor_element_count_ext(unsigned int n_dims, const unsigned int *shape){
    unsigned int size = 1;
    for(unsigned int i = 0; i < n_dims; i++)
        size *= shape[i];
    return size;
}

struct adci_multi_dim_counter adci_tensor_init_multidim_counter(const struct adci_tensor *tensor, const unsigned int *dims, unsigned int length){
    struct adci_multi_dim_counter counter = {.tensor = tensor, .free_dims_count = length};
    memcpy(counter.dim_indeces, dims, sizeof(unsigned int) * length);
    memset(counter.counter, 0, sizeof(counter.counter));
    for(unsigned int i = 0; i < tensor->n_dimension; i++)
        counter.precomputed_volumes[i] = adci_tensor_element_count_ext(tensor->n_dimension - i - 1, tensor->shape + i + 1);
    return counter;
}

struct adci_multi_dim_counter adci_tensor_alldim_counter_except(const struct adci_tensor *tensor, unsigned int excluded){
    struct adci_multi_dim_counter counter = {
        .tensor = tensor, 
        .free_dims_count = tensor->n_dimension - 1,
        .counter = {0}, 
    };
    unsigned int index = 0;
    for(unsigned int i = 0; i < tensor->n_dimension; i++){
        counter.precomputed_volumes[i] = adci_tensor_element_count_ext(tensor->n_dimension - i - 1, tensor->shape + i + 1);
        if(i == excluded) continue;
        counter.dim_indeces[index++] = i;
    }
    return counter;
}

void adci_tensor_increase_counter(struct adci_multi_dim_counter *counter){
    for(int i = (int)counter->free_dims_count - 1; i >= 0; i--){
        counter->counter[i]++;
        if(counter->counter[i] % counter->tensor->shape[counter->dim_indeces[i]] != 0) break;
        counter->counter[i] = 0;
    }
}

unsigned int adci_tensor_get_counter_offset(struct adci_multi_dim_counter counter){
    unsigned int offset = 0;
    for(unsigned int i = 0; i < counter.free_dims_count; i++)
        offset += counter.counter[i] * counter.precomputed_volumes[counter.dim_indeces[i]];
    return offset;
}