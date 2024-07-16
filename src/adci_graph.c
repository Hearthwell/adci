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

static struct adci_node * adci_graph_init_node(struct adci_graph *graph, struct adci_tensor *output){
    struct adci_node *node = ADCI_ALLOC(sizeof(struct adci_node));
    memset(node, 0, sizeof(struct adci_node));
    node->parents = adci_vector_init(sizeof(struct adci_node *));
    node->next = adci_vector_init(sizeof(struct adci_node *));
    node->inputs = adci_vector_init(sizeof(struct adci_tensor *));
    node->output = output;
    if(!output){
        node->output = ADCI_ALLOC(sizeof(struct adci_tensor));
        memset(node->output, 0, sizeof(struct adci_tensor));
    }
    adci_vector_add(&graph->nodes, &node);
    adci_vector_add(&graph->leafs, &node);
    adci_set_add(&graph->tensors, &node->output);
    return node;
}

static void adci_graph_handle_node_input(
    struct adci_graph *graph, 
    struct adci_node *node, 
    struct adci_graph_input current)
    {
    struct adci_tensor *tensor = current.input.tensor;
    if(current.type == ADCI_INPUT_NODE){
        adci_vector_add(&current.input.node->next, &node);
        adci_vector_add(&node->parents, &current.input.node);
        tensor = current.input.node->output;
        /* REMOVE NODE FROM LEAFS */
        /* TODO, MAYBE FIND MORE EFFICIENT WAY TO REMOVE */
        adci_vector_remove(&graph->leafs, &current.input.node);
    }else adci_set_add(&graph->tensors, &current.input.tensor);
    adci_vector_add(&node->inputs, &tensor);
}

static bool adci_exec_node(struct adci_node *node){
    if(node->op == ADCI_TENSOR_INPUT || node->computed) return true;
    for(unsigned int i = 0; i < node->parents.length; i++){
        struct adci_node *input = *(struct adci_node **)adci_vector_get(&node->parents, i);
        const bool tensor_ready = input->computed || input->op == ADCI_TENSOR_INPUT;
        if(!tensor_ready) return false;
    }
    /* ALL INPUTS ARE READY FOR EXECUTION OF CURRENT NODE */
    //printf("NODE TYPE: %s\n", adci_tensor_op_str(node->op));
    adci_tensor_compute_op(node->inputs, node->output, node->op);
    node->computed = true;
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
        adci_vector_free(&current->parents);
        adci_vector_free(&current->next);
        adci_vector_free(&current->inputs);
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
    struct adci_node *node = adci_graph_init_node(gf, output);
    node->op = op;
    /* MAKE SURE ALL INPUTS ARE LEAFS OR ADD NEW INPUTS */
    for(unsigned int i = 0; i < tensors.length; i++){
        struct adci_tensor *current = *(struct adci_tensor **)adci_vector_get(&tensors, i);
        /* IF TENSOR ALREADY EXISTS, IT WONT BE ADDED */
        adci_set_add(&gf->tensors, &current);
        adci_vector_add(&node->inputs, &current);
        struct adci_node *current_node = adci_graph_check_leaf(gf, current);
        /* CHECK IF NEW NODE OUTPUT CORRESPONDS TO OUTPUT OF EXISTING LEAF */
        if(current == output) adci_vector_remove(&gf->leafs, &current_node);
        if(current_node){
            /* ADD CURRENT NODE TO NEXT LIST */
            adci_vector_add(&current_node->next, &node);
            adci_vector_add(&node->parents, &current_node); 
            continue;
        }
        /* TENSOR NOT FOUND IN LEAFS, LOG AND ADD TENSOR TO HEAD (GRAPH INPUTS) */
        ADCI_LOG(ADCI_INFO, "Input tensor not a leaf, check inputs. Added as a graph input");
        struct adci_node *input = adci_graph_op_input(gf, current);
        adci_vector_add(&input->next, &node);
        adci_vector_add(&node->parents, &input);
    }
    return output;
}

struct adci_node * adci_graph_op_input(struct adci_graph *gf, struct adci_tensor *input){
    struct adci_node *node = adci_graph_init_node(gf, input);
    node->op = ADCI_TENSOR_INPUT;
    adci_vector_add(&gf->inputs, &node);
    return node;
}

struct adci_node * adci_graph_op_pad(struct adci_graph *gf, struct adci_node *node, uint32_t padding[][2]){
    struct adci_node *compute_node = adci_graph_init_node(gf, NULL);
    compute_node->op = ADCI_TENSOR_PAD;
    adci_graph_handle_node_input(gf, compute_node, adci_graph_op_input_node(node));
    struct adci_tensor *padding_tensor = adci_tensor_init_vargs(2, ADCI_I32, node->output->n_dimension, 2);    
    adci_tensor_alloc_set(padding_tensor, padding);
    adci_graph_handle_node_input(gf, compute_node, adci_graph_op_input_tensor(padding_tensor));
    adci_tensor_compute_op_shape(compute_node->inputs, compute_node->output, compute_node->op);
    return compute_node;
}

struct adci_node * adci_graph_op_conv2D(struct adci_graph *gf, struct adci_node *node, struct adci_graph_input filter, uint32_t stride[2], uint32_t dims[3]){
    struct adci_node *compute_node = adci_graph_init_node(gf, NULL);
    compute_node->op = ADCI_TENSOR_CONV2D;
    adci_graph_handle_node_input(gf, compute_node, adci_graph_op_input_node(node));
    adci_graph_handle_node_input(gf, compute_node, filter);
    struct adci_tensor *stride_tensor = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc_set(stride_tensor, stride);
    adci_graph_handle_node_input(gf, compute_node, adci_graph_op_input_tensor(stride_tensor));
    struct adci_tensor *dims_tensor = adci_tensor_init_vargs(1, ADCI_I32, 3);
    adci_tensor_alloc_set(dims_tensor, dims);
    adci_graph_handle_node_input(gf, compute_node, adci_graph_op_input_tensor(dims_tensor));
    adci_tensor_compute_op_shape(compute_node->inputs, compute_node->output, compute_node->op);
    return compute_node;
}

struct adci_node * adci_graph_op_max_pool2D(struct adci_graph *gf, struct adci_node *node, uint32_t size[2], uint32_t stride[2], uint32_t dims[2]){
    struct adci_node *compute_node = adci_graph_init_node(gf, NULL);
    compute_node->op = ADCI_TENSOR_MAX_POOL2D;
    adci_graph_handle_node_input(gf, compute_node, adci_graph_op_input_node(node));
    
    struct adci_tensor *size_tensor = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc_set(size_tensor, size);
    adci_graph_handle_node_input(gf, compute_node, adci_graph_op_input_tensor(size_tensor));
    
    struct adci_tensor *stride_tensor = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc_set(stride_tensor, stride);
    adci_graph_handle_node_input(gf, compute_node, adci_graph_op_input_tensor(stride_tensor));
    
    struct adci_tensor *dims_tensor = adci_tensor_init_vargs(1, ADCI_I32, 2);
    adci_tensor_alloc_set(dims_tensor, dims);
    adci_graph_handle_node_input(gf, compute_node, adci_graph_op_input_tensor(dims_tensor));
    adci_tensor_compute_op_shape(compute_node->inputs, compute_node->output, compute_node->op);
    return compute_node;
}

struct adci_node * adci_graph_op_relu(struct adci_graph *gf, struct adci_node *node){
    struct adci_node *compute_node = adci_graph_init_node(gf, NULL);
    compute_node->op = ADCI_TENSOR_RELU;
    adci_graph_handle_node_input(gf, compute_node, adci_graph_op_input_node(node));
    adci_tensor_compute_op_shape(compute_node->inputs, compute_node->output, compute_node->op);
    return compute_node;
}

struct adci_node * adci_graph_op_mul(struct adci_graph *gf, struct adci_node *node, struct adci_graph_input operand){
    struct adci_node *compute_node = adci_graph_init_node(gf, NULL);
    compute_node->op = ADCI_TENSOR_MUL;
    adci_graph_handle_node_input(gf, compute_node, adci_graph_op_input_node(node));
    adci_graph_handle_node_input(gf, compute_node, operand);
    adci_tensor_compute_op_shape(compute_node->inputs, compute_node->output, compute_node->op);
    return compute_node;
}

struct adci_node * adci_graph_op_add(struct adci_graph *gf, struct adci_node *node, struct adci_graph_input operand){
    struct adci_node *compute_node = adci_graph_init_node(gf, NULL);
    compute_node->op = ADCI_TENSOR_ADD;
    adci_graph_handle_node_input(gf, compute_node, adci_graph_op_input_node(node));
    adci_graph_handle_node_input(gf, compute_node, operand);
    adci_tensor_compute_op_shape(compute_node->inputs, compute_node->output, compute_node->op);
    return compute_node;
}

struct adci_node * adci_graph_op_sub(struct adci_graph *gf, struct adci_node *node, struct adci_graph_input operand){
    struct adci_node *compute_node = adci_graph_init_node(gf, NULL);
    compute_node->op = ADCI_TENSOR_SUB;
    adci_graph_handle_node_input(gf, compute_node, adci_graph_op_input_node(node));
    adci_graph_handle_node_input(gf, compute_node, operand);
    adci_tensor_compute_op_shape(compute_node->inputs, compute_node->output, compute_node->op);
    return compute_node;
}

struct adci_node * adci_graph_op_transpose(struct adci_graph *gf, struct adci_node *node, uint32_t dims[]){
    struct adci_node *compute_node = adci_graph_init_node(gf, NULL);
    compute_node->op = ADCI_TENSOR_TRANSPOSE;
    adci_graph_handle_node_input(gf, compute_node, adci_graph_op_input_node(node));
    struct adci_tensor *dims_tensor = adci_tensor_init_vargs(1, ADCI_I32, node->output->n_dimension);
    adci_tensor_alloc_set(dims_tensor, dims);
    adci_graph_handle_node_input(gf, compute_node, adci_graph_op_input_tensor(dims_tensor));
    adci_tensor_compute_op_shape(compute_node->inputs, compute_node->output, compute_node->op);
    return compute_node;
}

struct adci_node * adci_graph_op_reshape(struct adci_graph *gf, struct adci_node *node, struct adci_graph_input shape){
    struct adci_node *compute_node = adci_graph_init_node(gf, NULL);
    compute_node->op = ADCI_TENSOR_RESHAPE;
    adci_graph_handle_node_input(gf, compute_node, adci_graph_op_input_node(node));
    adci_graph_handle_node_input(gf, compute_node, shape);
    adci_tensor_compute_op_shape(compute_node->inputs, compute_node->output, compute_node->op);
    return compute_node;
}

struct adci_node * adci_graph_op_fully_connected(struct adci_graph *gf, struct adci_node *node, struct adci_graph_input weight){
    struct adci_node *compute_node = adci_graph_init_node(gf, NULL);
    compute_node->op = ADCI_TENSOR_FULLY_CONNECTED;
    adci_graph_handle_node_input(gf, compute_node, adci_graph_op_input_node(node));
    adci_graph_handle_node_input(gf, compute_node, weight);
    adci_tensor_compute_op_shape(compute_node->inputs, compute_node->output, compute_node->op);
    return compute_node;
}

struct adci_node * adci_graph_op_softmax(struct adci_graph *gf, struct adci_node *node, struct adci_graph_input dims){
    struct adci_node *compute_node = adci_graph_init_node(gf, NULL);
    compute_node->op = ADCI_TENSOR_SOFTMAX;
    adci_graph_handle_node_input(gf, compute_node, adci_graph_op_input_node(node));
    adci_graph_handle_node_input(gf, compute_node, dims);
    adci_tensor_compute_op_shape(compute_node->inputs, compute_node->output, compute_node->op);
    return compute_node;
}

struct adci_graph_input adci_graph_op_input_tensor(struct adci_tensor *tensor){
    return (struct adci_graph_input){.type = ADCI_INPUT_TENSOR, .input = {.tensor = tensor}};
}

struct adci_graph_input adci_graph_op_input_node(struct adci_node *node){
    return (struct adci_graph_input){.type = ADCI_INPUT_NODE, .input = {.node = node}};
}

struct adci_vector adci_graph_compute(struct adci_graph *gf){
    for(unsigned int i = 0; i < gf->inputs.length; i++){
        struct adci_node *current = *(struct adci_node **)adci_vector_get(&gf->inputs, i);
        adci_graph_compute_helper(current);
    }
    /* RESET COMPUTED STATE */
    for(unsigned int i = 0; i < gf->nodes.length; i++){
        struct adci_node *current = *(struct adci_node **)adci_vector_get(&gf->nodes, i);
        current->computed = false;
    }
    /* BUILD THE OUTPUT TENSORS VECTOR */
    struct adci_vector outputs = adci_vector_init(sizeof(struct adci_tensor *));
    for(unsigned int i = 0; i < gf->leafs.length; i++){
        struct adci_node *leaf = *(struct adci_node **)adci_vector_get(&gf->leafs, i);
        if(leaf->next.length > 0 || leaf->op == ADCI_TENSOR_INPUT) continue;
        adci_vector_add(&outputs, &leaf->output);
    }
    return outputs;
}