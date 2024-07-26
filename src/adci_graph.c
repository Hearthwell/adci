#include "adci_graph.h"
#include "adci_common.h"
#include "adci_logging.h"
#include "adci_tensor_op.h"

#define ADCI_PADDING_VALUE_U8 ((uint8_t) 0xff)
#define ADCI_SECTION_END_TOKEN_U32 ((uint32_t)0x6f7f8f9f)

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

static void adci_graph_dump_tensor(int fd, const struct adci_tensor *tensor){
    unsigned int current = ADCI_SEEK(fd, 0, SEEK_CUR);
    /* ADD PADDING FOR UINT32 ALLIGNMENT */
    for(unsigned int i = 0; i < current % sizeof(uint32_t); i++)
        write(fd, (uint8_t[]){ADCI_PADDING_VALUE_U8}, sizeof(uint8_t));
    /* WRITE TENSOR SIZE */
    write(fd, &tensor->n_dimension, sizeof(uint32_t));
    for(unsigned int i = 0; i < ADCI_TENSOR_MAX_DIM; i++){
        if(i < tensor->n_dimension) write(fd, tensor->shape + i, sizeof(uint32_t));
        else write(fd, (uint32_t[]){0}, sizeof(uint32_t));
    }
    /* WRITE TENSOR DTYPE */
    write(fd, (uint32_t[]){(uint32_t) tensor->dtype}, sizeof(uint32_t));

    /* WRITE TENSOR DATA */
    /* ALLIGN DATA, USEFUL FOR 64 BIT DTYPE IMPLEMENTATION */
    current = ADCI_SEEK(fd, 0, SEEK_CUR);
    const unsigned int element_size = adci_tensor_dtype_size(tensor->dtype);
    for(unsigned int i = 0; i < current % element_size; i++)
        write(fd, (uint8_t[]){ADCI_PADDING_VALUE_U8}, 1);
    write(fd, tensor->data, adci_tensor_element_count(tensor) * element_size);
}

static void adci_graph_dump_node(int fd, const struct adci_graph *graph, const struct adci_vector weight_tensors, const struct adci_node *node){
    /* ALLIGN FOR UINT32 */
    unsigned int current_offset = ADCI_SEEK(fd, 0, SEEK_CUR);
    for(unsigned int i = 0; i < current_offset % sizeof(uint32_t); i++)
        write(fd, (uint8_t[]){ADCI_PADDING_VALUE_U8}, 1);
    /* SAVE NODE TYPE */
    write(fd, (uint32_t[]){(uint32_t)node->op}, sizeof(uint32_t));
    /* SAVE THE NUMBER OF PARENT NODES */
    write(fd, (uint32_t[]){(uint32_t)node->parents.length}, sizeof(uint32_t));
    /* SAVE THE PARENT NODES INDECES */
    for(unsigned int i = 0; i < node->parents.length; i++){
        struct adci_node *current = *(struct adci_node **)adci_vector_get(&node->parents, i);
        /* GET THE INDEX OF THE NODE */
        /* TODO, SHOULD USE HASH-MAP INSTEAD OF LOOPING THROUGH THE NODES */
        const unsigned int index = adci_vector_find(&graph->nodes, &current);
        write(fd, (uint32_t[]){(uint32_t)index}, sizeof(uint32_t));
    }
    /* SAVE THE NUMBER OF WEIGHT TENSORS */
    write(fd, (uint32_t[]){(uint32_t)(node->inputs.length - node->parents.length)}, sizeof(uint32_t));
    /* SAVE THE TENSOR WEIGHTS (TENSORS NOT FROM PARENT NODES) INDECES IN CORRECT ORDER */
    for(unsigned int i = node->parents.length; i < node->inputs.length; i++){
        struct adci_tensor *current = *(struct adci_tensor **)adci_vector_get(&node->inputs, i);
        const unsigned int index = adci_vector_find(&weight_tensors, &current);
        write(fd, (uint32_t[]){(uint32_t)index}, sizeof(uint32_t));
    }
    /* SAVE SHAPE INFO ONLY IF INPUT TO GRAPH */
    if(node->op == ADCI_TENSOR_INPUT){
        write(fd, &node->output->n_dimension, sizeof(uint32_t));
        write(fd, node->output->shape, sizeof(node->output->shape));
    }
    /* NO NEED TO SAVE INFORMATION ABOUT NEXT TENSORS, SINCE WE SAVE THE PARENT NODES */
    
    /* FOR NOW, DONT ADD THE OUTPUT TENSOR INDEX TO GRAPH FILE */
    /* SO WILL NOT KEEP INFORMATION ABOUT SPECIAL REUSED TENSORS IN GENERIC API MODE */
}

static struct adci_tensor * adci_graph_parse_tensor(int fd){
    struct adci_tensor *tensor = ADCI_ALLOC(sizeof(struct adci_tensor));
    unsigned int current = ADCI_SEEK(fd, 0, SEEK_CUR);
    ADCI_SEEK(fd, current % sizeof(uint32_t), SEEK_CUR);
    read(fd, &tensor->n_dimension, sizeof(uint32_t));
    read(fd, tensor->shape, ADCI_TENSOR_MAX_DIM * sizeof(uint32_t));
    uint32_t tensor_dtype = 0;
    read(fd, &tensor_dtype, sizeof(uint32_t));
    tensor->dtype = (enum adci_tensor_type) tensor_dtype;
    const unsigned int data_size = adci_tensor_element_count(tensor) * adci_tensor_dtype_size(tensor->dtype); 
    tensor->data = ADCI_ALLOC(data_size);

    /* MOVE TO BEGINNING OF DATA */
    current = ADCI_SEEK(fd, 0, SEEK_CUR);
    ADCI_SEEK(fd, current % adci_tensor_dtype_size(tensor->dtype), SEEK_CUR);
    read(fd, tensor->data, data_size);
    return tensor;
}

/* THE NODES AND TENSORS ARE RELATIVE INDECES, NEED AN EXTRA STEP TO GET THE POINTERS */
struct adci_temp_node{
    enum adci_tensor_op op;
    uint32_t n_dimension;
    uint32_t shape[ADCI_TENSOR_MAX_DIM];
    struct adci_vector parents;
    struct adci_vector weight_tensors;
};
static struct adci_temp_node adci_graph_parse_node(int fd){
    unsigned int node_offset = ADCI_SEEK(fd, 0, SEEK_CUR);
    node_offset = ADCI_SEEK(fd, node_offset % sizeof(uint32_t), SEEK_CUR);
    struct adci_temp_node node = {0};
    node.parents = adci_vector_init(sizeof(uint32_t));
    node.weight_tensors = adci_vector_init(sizeof(uint32_t));
    uint32_t node_op = 0;
    read(fd, &node_op, sizeof(uint32_t));
    node.op = (enum adci_tensor_op)node_op;
    uint32_t num_parent_nodes = 0;
    read(fd, &num_parent_nodes, sizeof(uint32_t));
    for(unsigned int i = 0; i < num_parent_nodes; i++){
        uint32_t current = 0;
        read(fd, &current, sizeof(uint32_t));
        adci_vector_add(&node.parents, &current);
    }
    uint32_t num_weight_tensors = 0;
    read(fd, &num_weight_tensors, sizeof(uint32_t));
    for(unsigned int i = 0; i < num_weight_tensors; i++){
        uint32_t current = 0;
        read(fd, &current, sizeof(uint32_t));
        adci_vector_add(&node.weight_tensors, &current);
    }
    if(node.op == ADCI_TENSOR_INPUT){
        read(fd, &node.n_dimension, sizeof(uint32_t));
        read(fd, node.shape, sizeof(node.shape));
    }
    return node;
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

int adci_graph_dump(const struct adci_graph *graph, const char *path){
    int fd = ADCI_OPEN(path, O_WRONLY | O_CREAT, 0660);
    if(fd < 0){
        ADCI_LOG(ADCI_ERROR, "Could Not Open File: %s", path);
        return -1;
    }

    /* TODO, START BY DUMPING ALL THE META-DATA */
    /* WRITE END OF META-DATA TOKEN */
    write(fd, (uint32_t[]){ADCI_SECTION_END_TOKEN_U32}, sizeof(uint32_t));

    /* DUMP ALL THE NON-INPUT TENSOR WEIGHTS */
    struct adci_vector weight_tensors = adci_vector_init(sizeof(struct adci_tensor *));
    for(unsigned int i = 0; i < graph->nodes.length; i++){
        struct adci_node *node = *(struct adci_node **)adci_vector_get(&graph->nodes, i);
        /* THE FIRST VALUES OF INPUTS ARE ALWAYS OUTPUT TENSORS FROM PARENTS NODES */
        /* WE ONLY WANT TO COPY THE WEIGHTS TENSORS */
        for(unsigned int j = node->parents.length; j < node->inputs.length; j++){
            struct adci_tensor *current_weight = *(struct adci_tensor **)adci_vector_get(&node->inputs, j);
            adci_vector_add(&weight_tensors, &current_weight);
        }
    }

    /* DUMP THE NUMBER OF WEIGHT TENSORS */
    write(fd, (uint32_t[]){(uint32_t)weight_tensors.length}, sizeof(uint32_t));
    /* DUMP ACTUAL TENSORS */
    for(unsigned int i = 0; i < weight_tensors.length; i++){
        struct adci_tensor *current = *(struct adci_tensor **)adci_vector_get(&weight_tensors, i);
        adci_graph_dump_tensor(fd, current);
    }

    /* WRITE SEPERATOR BETWEEN TENSORS AND NODES */
    write(fd, (uint32_t[]){ADCI_SECTION_END_TOKEN_U32}, sizeof(uint32_t));

    /* WRITE NODE COUNT */
    write(fd, (uint32_t[]){(uint32_t)graph->nodes.length}, sizeof(uint32_t));

    /* DUMP ALL THE NODES */
    for(unsigned int i = 0; i < graph->nodes.length; i++){
        struct adci_node *node = *(struct adci_node **)adci_vector_get(&graph->nodes, i);
        adci_graph_dump_node(fd, graph, weight_tensors, node);
    }

    adci_vector_free(&weight_tensors);
    ADCI_CLOSE(fd);
    return 0;
}

struct adci_graph adci_graph_load(const char *path){
    int fd = ADCI_OPEN(path, O_RDONLY);
    int successful = true;
    if(fd < 0){
        ADCI_LOG(ADCI_ERROR, "COULD NOT OPEN GRAPH FILE: %s", path);
        return (struct adci_graph){0};
    }
    struct adci_graph graph = adci_graph_init();
    struct adci_vector weights = adci_vector_init(sizeof(struct adci_tensor *));
    uint32_t buffer = 0;
    read(fd, &buffer, sizeof(uint32_t));
    if(buffer != ADCI_SECTION_END_TOKEN_U32){
        ADCI_LOG(ADCI_ERROR, "WRONG GRAPH FILE FORMAT, WRONG START TOKEN");
        successful = false;
        goto cleanup;
    }
    uint32_t weight_count = 0;
    read(fd, &weight_count, sizeof(uint32_t));
    for(unsigned int i = 0; i < weight_count; i++){
        struct adci_tensor *current = adci_graph_parse_tensor(fd);
        adci_set_add(&graph.tensors, &current);
        adci_vector_add(&weights, &current);
    }
    read(fd, &buffer, sizeof(uint32_t));
    if(buffer != ADCI_SECTION_END_TOKEN_U32){
        ADCI_LOG(ADCI_ERROR, "WRONG SEPARATOR TOKEN BETWEEN TENSORS AND NODES");
        successful = false;
        goto cleanup;
    }
    uint32_t node_count = 0;
    read(fd, &node_count, sizeof(uint32_t));
    for(unsigned int i = 0; i < node_count; i++){
        struct adci_temp_node temp_node = adci_graph_parse_node(fd); 
        /* ALL PARENT NODES FOR CURRENT NODE SHOULD ALREADY HAVE BEEN PROCESSED */
        struct adci_node *current = adci_graph_init_node(&graph, NULL);
        current->op = temp_node.op;
        if(temp_node.op == ADCI_TENSOR_INPUT){
            current->output->n_dimension = temp_node.n_dimension;
            memcpy(current->output->shape, temp_node.shape, sizeof(temp_node.shape));
            adci_vector_add(&graph.inputs, &current);
        }
        for(unsigned int j = 0; j < temp_node.parents.length; j++){
            const uint32_t index = *(uint32_t *)adci_vector_get(&temp_node.parents, j);
            ADCI_ASSERT(index < graph.nodes.length);
            struct adci_node *parent = *(struct adci_node **)adci_vector_get(&graph.nodes, index);
            adci_vector_add(&parent->next, &current);
            adci_vector_add(&current->parents, &parent);
            adci_vector_add(&current->inputs, &parent->output);
        }
        for(unsigned int j = 0; j < temp_node.weight_tensors.length; j++){
            const uint32_t index = *(uint32_t *)adci_vector_get(&temp_node.weight_tensors, j);
            ADCI_ASSERT(index < weights.length);
            struct adci_tensor *weight = *(struct adci_tensor **)adci_vector_get(&weights, index);
            adci_vector_add(&current->inputs, &weight);
        }
        adci_tensor_compute_op_shape(current->inputs, current->output, current->op);
        adci_vector_free(&temp_node.parents);
        adci_vector_free(&temp_node.weight_tensors);
    }

    cleanup:
    adci_vector_free(&weights);
    ADCI_CLOSE(fd);
    if(!successful){
        adci_graph_free(&graph);
        return (struct adci_graph){0};
    }
    return graph;
}