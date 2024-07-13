#ifndef ADCI_GRAPH_H
#define ADCI_GRAPH_H

#include "adci_tensor.h"
#include "adci_common.h"
#include "adci_tensor_op.h"

struct adci_node{
    struct adci_vector parents; /* vector<adci_node *> */
    struct adci_vector inputs;  /* vector<adci_tensor *> */
    struct adci_tensor *output;
    enum adci_tensor_op op;
    /* OUTPUT CAN BE CONNECTED TO MULTIPLE NODES */
    struct adci_vector next;   /* vector<adci_node*> */
    bool computed;
};

struct adci_graph{
    struct adci_vector inputs;   /* vector<struct adci_node *> */
    struct adci_vector leafs;    /* vector<struct adci_node *> */
    struct adci_vector nodes;    /* vector<struct adci_node *> */
    struct adci_set    tensors;  /* set<struct adci_tensor *>  */
};

enum adci_graph_input_type{
    ADCI_INPUT_NODE,
    ADCI_INPUT_TENSOR
};
struct adci_graph_input{
    enum adci_graph_input_type type;
    union adci_graph_input_value{
        struct adci_node *node;
        struct adci_tensor *tensor;
    } input;
};

/* BUILD GRAPH USING TENSOR OPERATIONS */
struct adci_graph adci_graph_init();
void adci_graph_free(struct adci_graph *gf);

struct adci_string * adci_graph_str(const struct adci_graph *gf);

/* <GENERIC API> */
/* GENERIC GRAPH ADD NODE, INPUTS HAS TO BE IN FORM vector<struct adci_tensor *> */
/* YOU HAVE FULL CONTROL OF GRAPH LAYOUT THIS WAY, BE CAREFUL WHERE YOU OUTPUT YOUR RESULTS */
/* DONT MIX WITH THE OP API, BOTH MODES ARE NOT COMPATIBLE */
struct adci_tensor * adci_graph_add_node(struct adci_graph *gf, struct adci_vector tensors, struct adci_tensor *output, enum adci_tensor_op op);

/* <OP API> */
/* THIS IS AN EASIER AND FASTER TO USE API BUT WILL USE MORE MEMORY SINCE EACH NODE KEEPS IT'S OWN UNIQUE TENSOR OUTPUT */
/* AT LEAST ONE OF THE INPUTS HAVE TO BE A NODE, OTHERWISE GENERATED NODE WONT BE INTEGRATED INTO COMPUTE GRAPH */
/* TENSORS OWNERSHIP IS TRANSFERED, DONT DELETE TENSORS ANYMORE */
struct adci_node * adci_graph_op_input(struct adci_graph *gf, struct adci_tensor *input);
/* FIRST DIM OF PADDING MUST MATCH WITH tensor.n_dimension */
struct adci_node * adci_graph_op_pad(struct adci_graph *gf, struct adci_node *node, uint32_t padding[][2]);
struct adci_node * adci_graph_op_add(struct adci_graph *gf, struct adci_node *node, struct adci_graph_input operand);
struct adci_node * adci_graph_op_sub(struct adci_graph *gf, struct adci_node *node, struct adci_graph_input operand);

struct adci_graph_input adci_graph_op_input_tensor(struct adci_tensor *tensor);
struct adci_graph_input adci_graph_op_input_node(struct adci_node *node);

/* RETURN vector<struct adci_tensor *> */
struct adci_vector adci_graph_compute(struct adci_graph *gf);

#endif //ADCI_GRAPH_H