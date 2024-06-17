#include <stdio.h>

#include "adci_tensor.h"
#include "adci_graph.h"

int main(void){
    printf("HELLO ADCI\n");

    struct adci_tensor *a = adci_tensor_init(1, (unsigned int []){1}, ADCI_F32);
    struct adci_tensor *b = adci_tensor_init(1, (unsigned int []){1}, ADCI_F32);
    struct adci_tensor *result = adci_tensor_init(1, (unsigned int []){1}, ADCI_F32);

    struct adci_graph graph = adci_graph_init();

    /* DEFINE NEURAL NETWORK (COMPUTE GRAPH) LAYERS */
    {
        struct adci_vector tensors = adci_vector_from_array((struct adci_tensor *[]){a, b}, 2, sizeof(struct adci_tensor *));
        adci_graph_op_add(&graph, tensors, result);
        adci_vector_free(&tensors);
    }

    /* SET INPUTS / ALLOCATE TENSORS */
    adci_tensor_alloc(a);
    adci_tensor_alloc(b);
    adci_tensor_alloc(result);
    ((float *)a->data)[0] = 10.f;
    ((float *)b->data)[0] = 5.f;

    /* RUN GRAPH COMPUTATION */
    struct adci_vector outputs = adci_graph_compute(&graph);

    /* CHECK OUTPUT */
    float data = ((float *)(*(struct adci_tensor **)adci_vector_get(&outputs, 0))->data)[0];
    printf("OUTPUT: %f\n", data);

    /* CLEAN OUTPUTS VECTOR AND GRAPH */
    adci_vector_free(&outputs);
    adci_graph_free(&graph);

    return 0;
}