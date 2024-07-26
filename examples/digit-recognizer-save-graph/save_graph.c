#include <stdio.h>

#include "common.h"

#include "adci_graph.h"
#include "adci_image.h"

#include "../digit-recognizer-nograph/weights.h"

#define IMG_SIZE 28

int main(){
    printf("DIGIT RECOGNIZER MODEL WITH GRAPH IMPLEMENTATION\n");

    struct adci_graph graph = adci_graph_init();

    /* SETUP INPUTS */
    struct adci_tensor *input_tensor = adci_tensor_from_image("../digit-recognizer-nograph/inputs/img_10.jpg");
    struct adci_node *input = adci_graph_op_input(&graph, input_tensor);

    /* SETUP COMPUTE GRAPH */
    struct adci_node *padding_conv0 = adci_graph_op_pad(&graph, input, (uint32_t[][2]){{0, 0}, {1, 1}, {1, 1}, {0, 0}});
    
    struct adci_tensor *conv0_filter = adci_tensor_init_vargs(4, ADCI_F32, 32, 3, 3, 1);
    adci_tensor_alloc_set(conv0_filter, conv2D_filter);
    struct adci_node *conv0 = adci_graph_op_conv2D(&graph, padding_conv0, adci_graph_op_input_tensor(conv0_filter), (uint32_t[]){1, 1}, (unsigned int[]){1, 2, 3});
    
    struct adci_tensor *conv0_bias = adci_tensor_init_vargs(1, ADCI_F32, 32);
    adci_tensor_alloc_set(conv0_bias, conv2D_bias);
    struct adci_node *bias_conv0 = adci_graph_op_add(&graph, conv0, adci_graph_op_input_tensor(conv0_bias));

    struct adci_node *relu_conv0 = adci_graph_op_relu(&graph, bias_conv0);

    struct adci_tensor *const_mult_tensor = adci_tensor_init_vargs(1, ADCI_F32, 32);
    adci_tensor_alloc_set(const_mult_tensor, mult_weights);
    struct adci_node *const_mult = adci_graph_op_mul(&graph, relu_conv0, adci_graph_op_input_tensor(const_mult_tensor));

    struct adci_tensor *const_add_tensor = adci_tensor_init_vargs(1, ADCI_F32, 32);
    adci_tensor_alloc_set(const_add_tensor, add_weights);
    struct adci_node *const_add = adci_graph_op_add(&graph, const_mult, adci_graph_op_input_tensor(const_add_tensor));

    struct adci_node *padding_conv1 = adci_graph_op_pad(&graph, const_add, (uint32_t[][2]){{0, 0}, {1, 1}, {1, 1}, {0, 0}});
    
    struct adci_tensor *conv1_filter = adci_tensor_init_vargs(4, ADCI_F32, 64, 3, 3, 32);
    adci_tensor_alloc_set(conv1_filter, conv2D_filter_1);
    struct adci_node *conv1 = adci_graph_op_conv2D(&graph, padding_conv1, adci_graph_op_input_tensor(conv1_filter), (uint32_t[]){1, 1}, (unsigned int[]){1, 2, 3});
    
    struct adci_tensor *conv1_bias = adci_tensor_init_vargs(1, ADCI_F32, 64);
    adci_tensor_alloc_set(conv1_bias, conv2D_bias_1);
    struct adci_node *bias_conv1 = adci_graph_op_add(&graph, conv1, adci_graph_op_input_tensor(conv1_bias));
    
    struct adci_node *max_pool2D_0 = adci_graph_op_max_pool2D(&graph, bias_conv1, (uint32_t[]){2, 2}, (uint32_t[]){2, 2}, (uint32_t[]){1, 2});
    
    struct adci_node *padding_conv2 = adci_graph_op_pad(&graph, max_pool2D_0, (uint32_t[][2]){{0, 0}, {1, 1}, {1, 1}, {0, 0}});
    
    struct adci_tensor *conv2_filter = adci_tensor_init_vargs(4, ADCI_F32, 128, 3, 3, 64);
    adci_tensor_alloc_set(conv2_filter, conv2D_filter_2);
    struct adci_node *conv2 = adci_graph_op_conv2D(&graph, padding_conv2, adci_graph_op_input_tensor(conv2_filter), (uint32_t[]){1, 1}, (unsigned int[]){1, 2, 3});
    
    struct adci_tensor *conv2_bias = adci_tensor_init_vargs(1, ADCI_F32, 128);
    adci_tensor_alloc_set(conv2_bias, conv2D_bias_2);
    struct adci_node *bias_conv2 = adci_graph_op_add(&graph, conv2, adci_graph_op_input_tensor(conv2_bias));

    struct adci_node *max_pool2D_1 = adci_graph_op_max_pool2D(&graph, bias_conv2, (uint32_t[]){2, 2}, (uint32_t[]){2, 2}, (uint32_t[]){1, 2});

    struct adci_node *transpose_node = adci_graph_op_transpose(&graph, max_pool2D_1, (uint32_t[]){0, 3, 1, 2});

    struct adci_tensor *shape_tensor = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc_set(shape_tensor, (uint32_t[]){1, 6272});
    struct adci_node *reshape_node = adci_graph_op_reshape(&graph, transpose_node, adci_graph_op_input_tensor(shape_tensor));

    struct adci_tensor *connected_weights_0 = adci_tensor_init_vargs(2, ADCI_F32, 384, 6272);
    adci_tensor_alloc_set(connected_weights_0, fullyconnected_weights);
    struct adci_node *fully_connected = adci_graph_op_fully_connected(&graph, reshape_node, adci_graph_op_input_tensor(connected_weights_0));

    struct adci_tensor *connected_bias_0 = adci_tensor_init_vargs(1, ADCI_F32, 384);
    adci_tensor_alloc_set(connected_bias_0, fullyconnected_bias);
    struct adci_node *fully_connected_bias = adci_graph_op_add(&graph, fully_connected, adci_graph_op_input_tensor(connected_bias_0));

    struct adci_tensor *connected_weights_1 = adci_tensor_init_vargs(2, ADCI_F32, 10, 384);
    adci_tensor_alloc_set(connected_weights_1, fullyconnected_weights_1);
    struct adci_node *fully_connected_1 = adci_graph_op_fully_connected(&graph, fully_connected_bias, adci_graph_op_input_tensor(connected_weights_1));

    struct adci_tensor *connected_bias_1 = adci_tensor_init_vargs(1, ADCI_F32, 10);
    adci_tensor_alloc_set(connected_bias_1, fullyconnected_bias_1);
    struct adci_node *fully_connected_bias_1 = adci_graph_op_add(&graph, fully_connected_1, adci_graph_op_input_tensor(connected_bias_1));

    struct adci_tensor *softmax_dims = adci_tensor_init_vargs(1, ADCI_I32, 1);
    adci_tensor_alloc_set(softmax_dims, (uint32_t[]){1});
    adci_graph_op_softmax(&graph, fully_connected_bias_1, adci_graph_op_input_tensor(softmax_dims));

    /* SAVE GRAPH */
    int status = adci_graph_dump(&graph, DIGIT_RECOGNIZER_GRAPH);

    /* CLEAN GRAPH */
    adci_graph_free(&graph);

    if(status != 0){
        printf("ERROR WHILE SAVING GRAPH\n");
        return 1;
    }

    return 0;
}