# ADCI

## Cpu Inference Engine For Neural Networks

### Introduction

#### Neural network inference library for embedded systems. Focus on simplicity and efficiency (STILL WORK UNDER PROGRESS). No dependencies, only thing needed is a C compiler.

### Build
```
# FOR DEBUG BUILD
make lib

# FOR RELEASE BUILD
make release_build
```

### Getting Started

#### This Example shows how to build a simple add Graph
```
#include "adci_graph.h"

int main(){
    struct adci_graph graph = adci_graph_init();

    struct adci_tensor *first = adci_tensor_init_vargs(4, ADCI_F32, 32, 3, 3, 1);
    adci_tensor_alloc(first);
    /* FILL TENSOR WITH DATA */

    struct adci_node *input = adci_graph_op_input(&graph, first);

    struct adci_tensor *second = adci_tensor_init_vargs(4, ADCI_F32, 32, 3, 3, 1);
    adci_tensor_alloc(second);
    /* FILL TENSOR WITH DATA */

    struct adci_node *add_node = adci_graph_op_add(&graph, input, adci_graph_op_input_tensor(second));

    /* SINCE THE SECOND TENSOR IS BEING ADDED AS A TENSOR AND NOT A NODE, IT SHOULD BE REGARDED AS A WEIGHT AND NOT AN INPUT TO THE NETWORK */

    /* RUN ACTUAL COMPUTATION */
    struct adci_vector outputs = adci_graph_compute(&graph);
    struct adci_tensor *logits = *(struct adci_tensor **)adci_vector_get(&outputs, 0);

    /* CLEAN GRAPH */
    adci_vector_free(&outputs);
    adci_graph_free(&graph);

    return 0;
}
```

### RoadMap

- [x] Add Basic Tensor operations for simple neural network
- [x] Implement basic neural network as example (mnist digit recognizer)
- [ ] Add support for importing model from onnx format
- [ ] Add support for importing model from tensorflow-lite format
- [ ] Add support for importing pytorch model weights
- [x] Add support for saving/loading compute graph and network weigths into/from file
- [ ] Add Multiple Thread support to tensor operations 
- [ ] Implement Ultralytics YOLOV8 model
- [ ] Implement TwinLiteNet model
- [ ] Add Missing Tensor operations for LLM/transformer models support
- [ ] Implement OpenAi Whisker model
- [ ] Implement Llama model 

### Examples

- [MNIST digit recognizer without using graph api (bare tensor operation)](./examples/digit-recognizer-nograph/)

- [MNIST digit recognizer using graph api (much simpler and recommended)](./examples/digit-recognizer/)

