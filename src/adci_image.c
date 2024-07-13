#include "external/stb_image.h"
#include "adci_tensor_op.h"

#include "stb_image.h"

/* RETURNS TENSOR IN FORMAT BWHC */
struct adci_tensor * adci_tensor_from_image(const char *path){
    /* BATCH VALUE SET TO 1 */
    int shape[4] = {1, 0, 0, 0};
    unsigned char *data = stbi_load(path, shape + 1, shape + 2, shape + 3, 0);
    struct adci_tensor *tensor = adci_tensor_init(sizeof(shape) / sizeof(unsigned int), (unsigned int *)shape, ADCI_F32);
    adci_tensor_alloc(tensor);
    const unsigned int element_count = adci_tensor_element_count(tensor);
    for(unsigned int i = 0; i < element_count; i++)
        ((float *)tensor->data)[i] = (float)(data[i]);
    ADCI_FREE(data);
    return tensor;
}