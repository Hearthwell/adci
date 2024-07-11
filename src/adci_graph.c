#include "adci_graph.h"
#include "adci_common.h"
#include "adci_logging.h"
#include "adci_tensor_op.h"

/* PRIVATE FUNCTIONS */

static struct adci_node * adci_graph_check_leaf(const struct adci_graph *gf, const struct adci_tensor *tensor){
    for(unsigned int i = 0; i < gf->leafs.length; i++){
        struct adci_node *current = *(struct adci_node **)adci_vector_get(&gf->leafs, i);
        if(current->output == tensor) return current;
    }
    return NULL;
}

static struct adci_node * adci_graph_add_input_node(struct adci_graph *graph, struct adci_tensor *input){
    struct adci_node *node = ADCI_ALLOC(sizeof(struct adci_node));
    memset(&node->inputs, 0, sizeof(struct adci_vector));
    node->output = input;
    node->op = ADCI_TENSOR_INPUT;
    node->next = adci_vector_init(sizeof(struct adci_node *));
    adci_vector_add(&graph->nodes, &node);
    return node;
}

static bool adci_exec_node(struct adci_node *node){
    if(node->op == ADCI_TENSOR_INPUT || node->computed) return true;
    for(unsigned int i = 0; i < node->inputs.length; i++){
        struct adci_node *input = *(struct adci_node **)adci_vector_get(&node->inputs, i);
        const bool tensor_ready = input->computed || input->op == ADCI_TENSOR_INPUT;
        if(!tensor_ready) return false;
    }
    /* ALL INPUTS ARE READY FOR EXECUTION OF CURRENT NODE */
    struct adci_vector tensors = adci_vector_init(sizeof(struct adci_tensor *));
    for(unsigned int i = 0; i < node->inputs.length; i++){
        struct adci_node *input = *(struct adci_node **)adci_vector_get(&node->inputs, i);
        adci_vector_add(&tensors, &input->output);
    }
    adci_tensor_compute_op(tensors, node->output, node->op);
    adci_vector_free(&tensors);
    return true;
}

static void adci_graph_compute_helper(struct adci_node *root){
    if(!adci_exec_node(root)) return;
    for(unsigned int i = 0; i < root->next.length; i++){
        struct adci_node *current = *(struct adci_node **)adci_vector_get(&root->next, i);
        bool computed = adci_exec_node(current);
        if(!computed) continue;
        adci_graph_compute_helper(current);
    }
}

/* END PRIVATE FUNCTIONS */

struct adci_graph adci_graph_init(){
    struct adci_graph graph = {0};
    graph.nodes = adci_vector_init(sizeof(struct adci_node *));
    graph.inputs = adci_vector_init(sizeof(struct adci_node *));
    graph.leafs  = adci_vector_init(sizeof(struct adci_node *));
    graph.tensors = adci_set_init(sizeof(struct adci_tensor *), NULL);
    return graph;
}

void adci_graph_free(struct adci_graph *gf){
    /* CLEAN TENSORS */
    struct adci_set_iterator iterator = adci_set_get_iterator(&gf->tensors);
    while(!iterator.done){
        struct adci_tensor **tensor = adci_set_get_next(&iterator);
        if(tensor) adci_tensor_free(*tensor);
    }
    /* CLEAN NODES */
    for(unsigned int i = 0; i < gf->nodes.length; i++){
        struct adci_node *current = *(struct adci_node **)adci_vector_get(&gf->nodes, i);
        adci_vector_free(&current->inputs);
        adci_vector_free(&current->next);
        ADCI_FREE(current);
    }
    /* CLEAN CONTAINERS */
    adci_set_free(&gf->tensors);
    adci_vector_free(&gf->leafs);
    adci_vector_free(&gf->inputs);
    adci_vector_free(&gf->nodes);
}

struct adci_string * adci_graph_str(const struct adci_graph *gf){
    (void)gf;
    ADCI_ASSERT("TODO, NOT IMPLEMENTED" == 0);
    return NULL;
}

struct adci_tensor * adci_graph_add_node(struct adci_graph *gf, struct adci_vector tensors, struct adci_tensor *output, enum adci_tensor_op op){
    struct adci_node *node = ADCI_ALLOC(sizeof(struct adci_node));
    adci_vector_add(&gf->nodes, &node);
    node->op = op;
    node->inputs = adci_vector_init(sizeof(struct adci_node *));
    node->next = adci_vector_init(sizeof(struct adci_node *));
    /* MAKE SURE ALL INPUTS ARE LEAFS OR ADD NEW INPUTS */
    for(unsigned int i = 0; i < tensors.length; i++){
        struct adci_tensor *current = *(struct adci_tensor **)adci_vector_get(&tensors, i);
        /* IF TENSOR ALREADY EXISTS, IT WONT BE ADDED */
        adci_set_add(&gf->tensors, &current);
        struct adci_node *current_node = adci_graph_check_leaf(gf, current);
        if(current_node){
            /* ADD CURRENT NODE TO NEXT LIST */
            adci_vector_add(&current_node->next, &node);
            adci_vector_add(&node->inputs, &current_node); 
            continue;
        }
        /* TENSOR NOT FOUND IN LEAFS, LOG AND ADD TENSOR TO HEAD (GRAPH INPUTS) */
        ADCI_LOG(ADCI_INFO, "Input tensor not a leaf, check inputs. Added as a graph input");
        struct adci_node *input = adci_graph_add_input_node(gf, current);
        adci_vector_add(&input->next, &node);
        adci_vector_add(&node->inputs, &input); 
        adci_vector_add(&gf->inputs, &input);
        adci_vector_add(&gf->leafs, &input);
    }
    node->output = output;
    adci_set_add(&gf->tensors, &output);
    adci_vector_add(&gf->leafs, &node);
    return output;
}

struct adci_vector adci_graph_compute(struct adci_graph *gf){
    for(unsigned int i = 0; i < gf->inputs.length; i++){
        struct adci_node *current = *(struct adci_node **)adci_vector_get(&gf->inputs, i);
        adci_graph_compute_helper(current);
    }
    /* BUILD THE OUTPUT TENSORS VECTOR */
    struct adci_vector outputs = adci_vector_init(sizeof(struct adci_tensor *));
    for(unsigned int i = 0; i < gf->leafs.length; i++){
        struct adci_node *leaf = *(struct adci_node **)adci_vector_get(&gf->leafs, i);
        if(leaf->next.length > 0) continue;
        adci_vector_add(&outputs, &leaf->output);
    }
    return outputs;
}