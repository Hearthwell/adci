#include "adci_common.h"

#define DEFAULT_VECT_CAPACITY 10

struct adci_string * adci_init_str(const char *buffer, unsigned int length){
    struct adci_string *str = (struct adci_string *)ADCI_ALLOC(sizeof(struct adci_string));
    str->size = length;
    str->str = ADCI_ALLOC(length + 1);
    memcpy(str->str, buffer, length);
    str->str[length] = '\0';
    return str;
}

bool adci_clean_str(struct adci_string *str){
    ADCI_FREE(str->str);
    str->str = NULL;
    ADCI_FREE(str);
    return true;
}

struct adci_vector adci_vector_init(unsigned int element_bsize){
    struct adci_vector vect = {0};
    vect.bsize = element_bsize;
    vect.capacity = DEFAULT_VECT_CAPACITY;
    vect.data = ADCI_ALLOC(vect.capacity * vect.bsize);
    vect.length = 0;
    return vect;
}

bool adci_vector_add(struct adci_vector *vector, const void *element){
    if(vector->length == vector->capacity){
        vector->capacity *= 2;
        vector->data = ADCI_REALLOC(vector->data, vector->capacity);
    }
    memcpy((uint8_t*)vector->data + vector->length * vector->bsize, element, vector->bsize);
    vector->length++;
    return true;
}

bool adci_vector_remove(struct adci_vector *vector, const void *element){
    for(unsigned int i = 0; i < vector->length; i++){
        if(memcmp((uint8_t *)vector->data + i * vector->bsize, element, vector->bsize) != 0)
            continue;
        const unsigned int copy_size = (vector->length - i - 1) * vector->bsize;
        memcpy((uint8_t *)vector->data + i * vector->bsize, (uint8_t *)vector->data + (i + 1) * vector->bsize, copy_size);
        vector->length--;
        return true;
    }
    return false;
}

void * adci_vector_get(struct adci_vector *vector, unsigned int index){
    return (uint8_t *)vector->data + index * vector->bsize;
}

bool adci_vector_free(struct adci_vector *vector){
    ADCI_FREE(vector->data);
    vector->data = NULL;
    return true;
}