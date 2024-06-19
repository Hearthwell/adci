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

/* TENSORS OWNERSHIP IS TRANSFERED, DONT DELETE TENSORS ANYMORE OR VECTOR */
struct adci_tensor * adci_graph_op_add (struct adci_graph *gf, struct adci_vector tensors, struct adci_tensor *output);
struct adci_tensor * adci_graph_op_sub (struct adci_graph *gf, struct adci_vector tensors, struct adci_tensor *output);
struct adci_tensor * adci_graph_op_copy(struct adci_graph *gf, struct adci_tensor *tensor, struct adci_tensor *output);

/* TODO, IMPLEMENT WHEN WE HAVE A WAY TO SAVE ENTIRE GRAPH INFO IN A FILE */
//bool adci_graph_allocate_tensors(struct adci_graph *gf);
//bool adci_graph_deallocate_tensors(struct adci_graph *gf);

/* RETURN vector<struct adci_tensor *> */
struct adci_vector adci_graph_compute(struct adci_graph *gf);

#endif //ADCI_GRAPH_H