#include "common.h"

#include "adci_graph.h"
#include "adci_image.h"

int main(){
    printf("DIGIT-RECOGNIZER LOAD GRAPH EXAMPLE\n");

    struct adci_graph graph = adci_graph_load(DIGIT_RECOGNIZER_GRAPH);

    struct adci_tensor *img = adci_tensor_from_image("../digit-recognizer-nograph/inputs/img_10.jpg");
    struct adci_node *input = *(struct adci_node **)adci_vector_get(&graph.inputs, 0);
    adci_tensor_alloc(input->output);
    memcpy(input->output->data, img->data, adci_tensor_element_count(input->output) * adci_tensor_dtype_size(input->output->dtype));
    adci_tensor_free(img);

    struct adci_vector outputs = adci_graph_compute(&graph);
    struct adci_tensor *logits = *(struct adci_tensor **)adci_vector_get(&outputs, 0);

    printf("LENGTH: %d\n", outputs.length);
    adci_tensor_print_shape(logits);

    /* ARGMAX WOULD BE NICE AT THIS POINT */
    float max = ((float *)logits->data)[0];
    unsigned int index = 0;
    for(unsigned int i = 0; i < logits->shape[1]; i++){
        printf("Index: %d, probability: %f\n", i, ((float *)logits->data)[i]);
        if(((float *)logits->data)[i] < max) continue; 
        max = ((float *)logits->data)[i];
        index = i;
    }

    printf("INDEX: %d\n", index);

    /* CLEAN UP */
    adci_vector_free(&outputs);
    adci_graph_free(&graph);

    return 0;
}