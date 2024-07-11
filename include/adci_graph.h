#ifndef ADCI_GRAPH_H
#define ADCI_GRAPH_H

#include "adci_tensor.h"
#include "adci_common.h"
#include "adci_tensor_op.h"

struct adci_node{
    struct adci_vector inputs; /* vector<adci_node*> */
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

/* BUILD GRAPH USING TENSOR OPERATIONS */
struct adci_graph adci_graph_init();
void adci_graph_free(struct adci_graph *gf);

struct adci_string * adci_graph_str(const struct adci_graph *gf);

/* GENERIC GRAPH ADD NODE, INPUTS HAS TO BE IN FORM vector<struct adci_tensor *> */
struct adci_tensor * adci_graph_add_node(struct adci_graph *gf, struct adci_vector tensors, struct adci_tensor *output, enum adci_tensor_op op);

/* TENSORS OWNERSHIP IS TRANSFERED, DONT DELETE TENSORS ANYMORE OR VECTOR */

/* RETURN vector<struct adci_tensor *> */
struct adci_vector adci_graph_compute(struct adci_graph *gf);

#endif //ADCI_GRAPH_H