#include <gtest/gtest.h>

extern "C"{
#include "adci_graph.h"
#include "adci_tensor_op.h"
}

#define ADCI_GRAPH_SUITE_NAME ADCI_GRAPH

TEST(ADCI_GRAPH_SUITE_NAME, adci_graph_init){
    struct adci_graph graph = adci_graph_init();
    EXPECT_EQ(graph.inputs.length, 0);
    EXPECT_EQ(graph.leafs.length, 0);
    EXPECT_EQ(graph.nodes.length, 0);
    EXPECT_EQ(graph.tensors.length, 0);
    adci_graph_free(&graph);
}

TEST(ADCI_GRAPH_SUITE_NAME, adci_graph_op_add){
    unsigned int shape[] = {10, 10};
    struct adci_graph graph = adci_graph_init();
    adci_tensor *a = adci_tensor_init_2d(shape[0], shape[1], ADCI_F32);
    adci_tensor *b = adci_tensor_init_2d(shape[0], shape[1], ADCI_F32);
    adci_tensor *output = adci_tensor_init_2d(shape[0], shape[1], ADCI_F32);
    adci_tensor *inputs[] = {a, b};
    adci_vector tensors = adci_vector_from_array(inputs, 2, sizeof(adci_tensor *));
    adci_graph_op_add(&graph, tensors, output);
    adci_vector_free(&tensors);
    EXPECT_EQ(graph.inputs.length, 2);
    EXPECT_EQ(graph.leafs.length, 3);
    EXPECT_EQ(graph.nodes.length, 3);
    EXPECT_EQ(graph.tensors.length, 3);
    adci_node *add_node = nullptr;
    for(unsigned int i = 0; i < graph.nodes.length; i++){
        add_node = *(adci_node **)adci_vector_get(&graph.nodes, i);
        if(add_node->op == ADCI_TENSOR_ADD) break;
    }
    EXPECT_EQ(add_node->inputs.length, 2);
    EXPECT_EQ(add_node->next.length, 0);
    EXPECT_EQ(add_node->output, output);
    adci_graph_free(&graph);
}

TEST(ADCI_GRAPH_SUITE_NAME, adci_graph_op_compute){
    unsigned int shape[] = {10, 10};
    struct adci_graph graph = adci_graph_init();
    adci_tensor *a = adci_tensor_init_2d(shape[0], shape[1], ADCI_F32);
    adci_tensor *b = adci_tensor_init_2d(shape[0], shape[1], ADCI_F32);
    adci_tensor *output = adci_tensor_init_2d(shape[0], shape[1], ADCI_F32);
    adci_tensor_alloc(a);
    adci_tensor_alloc(b);
    adci_tensor_alloc(output);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++){
        ((float *)a->data)[i] = (float)i;
        ((float *)b->data)[i] = (float)(shape[0] * shape[1] - i);
    }
    adci_tensor *inputs[] = {a, b};
    adci_vector tensors = adci_vector_from_array(inputs, 2, sizeof(adci_tensor *));
    adci_graph_op_add(&graph, tensors, output);
    adci_vector_free(&tensors);
    adci_vector outputs = adci_graph_compute(&graph);
    EXPECT_EQ(outputs.length, 1);
    EXPECT_EQ(*(adci_tensor **)adci_vector_get(&outputs, 0), output);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++){
        EXPECT_FLOAT_EQ(((float *)output->data)[i], shape[0] * shape[1]);
    }
    adci_vector_free(&outputs);
    adci_graph_free(&graph);
}