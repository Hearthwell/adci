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

TEST(ADCI_GRAPH_SUITE_NAME, adci_graph_add_node){
    unsigned int shape[] = {10, 10};
    struct adci_graph graph = adci_graph_init();
    adci_tensor *a = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor *b = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor *output = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor *inputs[] = {a, b};
    adci_vector tensors = adci_vector_from_array(inputs, 2, sizeof(adci_tensor *));
    adci_graph_add_node(&graph, tensors, output, ADCI_TENSOR_ADD);
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
    adci_tensor *a = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor *b = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor *output = adci_tensor_init_vargs(2, ADCI_F32, shape[0], shape[1]);
    adci_tensor_alloc(a);
    adci_tensor_alloc(b);
    adci_tensor_alloc(output);
    for(unsigned int i = 0; i < shape[0] * shape[1]; i++){
        ((float *)a->data)[i] = (float)i;
        ((float *)b->data)[i] = (float)(shape[0] * shape[1] - i);
    }
    adci_tensor *inputs[] = {a, b};
    adci_vector tensors = adci_vector_from_array(inputs, 2, sizeof(adci_tensor *));
    adci_graph_add_node(&graph, tensors, output, ADCI_TENSOR_ADD);
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

TEST(ADCI_GRAPH_SUITE_NAME, adci_graph_op_add){
    struct adci_graph graph = adci_graph_init();
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, 4, 4);
    adci_tensor_alloc(first);
    float value = 1.f;
    adci_tensor_fill(first, &value);
    adci_tensor *second = adci_tensor_init_vargs(2, ADCI_F32, 1, 4);
    adci_tensor_alloc(second);
    adci_tensor_fill(second, &value);
    adci_node *node = adci_graph_op_input(&graph, first);
    adci_graph_op_add(&graph, node, adci_graph_op_input_tensor(second));
    EXPECT_EQ(graph.inputs.length, 1);
    EXPECT_EQ(graph.leafs.length, 1);
    adci_graph_free(&graph);
}

TEST(ADCI_GRAPH_SUITE_NAME, adci_graph_op_add_compute){
    struct adci_graph graph = adci_graph_init();
    adci_tensor *first = adci_tensor_init_vargs(2, ADCI_F32, 4, 4);
    adci_tensor_alloc(first);
    float value = 1.f;
    adci_tensor_fill(first, &value);
    adci_tensor *second = adci_tensor_init_vargs(2, ADCI_F32, 1, 4);
    adci_tensor_alloc(second);
    adci_tensor_fill(second, &value);
    adci_node *node = adci_graph_op_input(&graph, first);
    adci_graph_op_add(&graph, node, adci_graph_op_input_tensor(second));
    adci_vector outputs = adci_graph_compute(&graph);
    EXPECT_EQ(outputs.length, 1);
    adci_tensor *output = *((adci_tensor **)adci_vector_get(&outputs, 0));
    EXPECT_EQ(output->shape[0], 4);
    EXPECT_EQ(output->shape[1], 4);
    for(unsigned int i = 0; i < 16; i++)
        EXPECT_EQ(((float *)output->data)[i], 2.f);
    adci_vector_free(&outputs);
    adci_graph_free(&graph);
}

TEST(ADCI_GRAPH_SUITE_NAME, adci_graph_dump_load){
    struct adci_graph graph = adci_graph_init();
    struct adci_node *first = adci_graph_op_input(&graph, adci_tensor_init_vargs(2, ADCI_F32, 4, 4));
    struct adci_node *second = adci_graph_op_input(&graph, adci_tensor_init_vargs(2, ADCI_F32, 4, 4));
    adci_graph_op_add(&graph, first, adci_graph_op_input_node(second));
    int status = adci_graph_dump(&graph, "out/test.adci");
    EXPECT_EQ(status, 0);
    adci_graph_free(&graph);
    struct adci_graph loaded = adci_graph_load("out/test.adci");
    EXPECT_EQ(loaded.inputs.length, 2);
    EXPECT_EQ(loaded.nodes.length, 3);
    adci_node *add_node = *(adci_node **)adci_vector_get(&loaded.nodes, 2); 
    EXPECT_EQ(add_node->op, ADCI_TENSOR_ADD);
    EXPECT_EQ(loaded.tensors.length, 3);
    adci_graph_free(&loaded);
}

TEST(ADCI_GRAPH_SUITE_NAME, adci_graph_dump_load_execute){
    struct adci_graph graph = adci_graph_init();
    struct adci_node *first = adci_graph_op_input(&graph, adci_tensor_init_vargs(2, ADCI_F32, 4, 4));
    struct adci_node *second = adci_graph_op_input(&graph, adci_tensor_init_vargs(2, ADCI_F32, 4, 4));
    adci_graph_op_add(&graph, first, adci_graph_op_input_node(second));
    int status = adci_graph_dump(&graph, "out/test.adci");
    EXPECT_EQ(status, 0);
    adci_graph_free(&graph);
    struct adci_graph loaded = adci_graph_load("out/test.adci");
    const float value = 1.f;
    for(unsigned int i = 0; i < loaded.inputs.length; i++){
        struct adci_node *input = *(struct adci_node **)adci_vector_get(&loaded.inputs, i);
        adci_tensor_alloc(input->output);
        adci_tensor_fill(input->output, &value);
    }
    struct adci_vector outputs = adci_graph_compute(&loaded);
    EXPECT_EQ(outputs.length, 1);
    struct adci_tensor *output = *(struct adci_tensor **)adci_vector_get(&outputs, 0); 
    for(unsigned int i = 0; i < adci_tensor_element_count(output); i++)
        EXPECT_FLOAT_EQ(((float *)output->data)[i], 2 * value);
    adci_vector_free(&outputs);
    adci_graph_free(&loaded);
}

TEST(ADCI_GRAPH_SUITE_NAME, adci_graph_compute_GENERIC_API){
    /* TODO, IMPLEMENT */
}