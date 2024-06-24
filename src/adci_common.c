#include "adci_common.h"

#define DEFAULT_VECT_CAPACITY 10
#define DEFAULT_SET_CAPACITY 10
#define SET_COLLISION_TRESHOLD 5

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

struct adci_vector adci_vector_from_array(void *elements, unsigned int count, unsigned int element_bsize){
    struct adci_vector vect = {0};
    vect.bsize = element_bsize;
    vect.capacity = (count < DEFAULT_VECT_CAPACITY) ? DEFAULT_VECT_CAPACITY : count;
    vect.data = ADCI_ALLOC(vect.capacity * vect.bsize);
    vect.length = count;
    memcpy(vect.data, elements, vect.length * vect.bsize);
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

void * adci_vector_get(const struct adci_vector *vector, unsigned int index){
    return (uint8_t *)vector->data + index * vector->bsize;
}

bool adci_vector_has(const struct adci_vector *vector, const void *element){
    for(unsigned int i = 0; i < vector->length; i++){
        if(memcmp(vector->data + i * vector->bsize, element, vector->bsize) == 0) return true;
    }
    return false;
}

void adci_vector_free(struct adci_vector *vector){
    if(!vector->data) return;
    ADCI_FREE(vector->data);
    vector->data = NULL;
}

/* SET IMPLEMENTATION */
struct adci_set_node{
    void *data;
    struct adci_set_node *next;
};

/* PRIVATE FUNCTIONS */

static unsigned int adci_set_defaut_hasher(const struct adci_set *set, const void *data){
    unsigned long hash = 5381;
    for(unsigned int i = 0; i < set->bsize; i++)
        hash = ((hash << 5) + hash) + ((uint8_t*)data)[i];
    return hash;
}

static void adci_set_copy_data(struct adci_set *previous, struct adci_set *set);

static bool adci_set_add_node(struct adci_set *set, struct adci_set_node *current){
    unsigned int index = set->hasher(set, current->data) % set->capacity;
    struct adci_set_node *node = set->data[index];
    unsigned int depth = 0;
    while(node && node->next != NULL && depth < SET_COLLISION_TRESHOLD){
        if(memcmp(node->data, current->data, set->bsize) == 0) return false;
        node = node->next;
        depth++;
    }
    if(depth >= SET_COLLISION_TRESHOLD){
        struct adci_set previous = *set;
        set->capacity *= 2;
        set->data = ADCI_ALLOC(set->capacity * sizeof(struct adci_set_node *));
        set->length = 0;
        adci_set_copy_data(&previous, set);
        ADCI_FREE(previous.data);
        return adci_set_add_node(set, current);
    }
    if(!node) set->data[index] = current;
    else node->next = current;
    set->length++;
    return true;
}

static void adci_set_copy_data(struct adci_set *previous, struct adci_set *set){
    for(unsigned int i = 0; i < previous->capacity; i++){
        struct adci_set_node *current = previous->data[i];
        while(current){
            struct adci_set_node *temp = current;
            current = current->next;
            temp->next = NULL;
            adci_set_add_node(set, temp);
        }
    }
}

/* END PRIVATE FUNCTIONS */

adci_set_hash adci_set_get_default_hasher(){
    return adci_set_defaut_hasher;
}

struct adci_set adci_set_init(unsigned int element_bsize, adci_set_hash hasher){
    struct adci_set set = {.bsize = element_bsize, .hasher = hasher, .length = 0};
    if(hasher == NULL) set.hasher = adci_set_defaut_hasher;
    set.capacity = DEFAULT_SET_CAPACITY;
    const unsigned int size = set.capacity * sizeof(struct adci_set_node *);
    set.data = ADCI_ALLOC(size);
    memset(set.data, 0, size);
    return set;
}

void adci_set_free(struct adci_set *set){
    for(unsigned int i = 0; i < set->capacity; i++){
        struct adci_set_node *current = set->data[i];
        while(current != NULL){
            struct adci_set_node *temp = current;
            current = current->next;
            ADCI_FREE(temp->data);
            ADCI_FREE(temp);
        }
    }
    ADCI_FREE(set->data);
    set->data = NULL;
}

bool adci_set_add(struct adci_set *set, const void *element){
    struct adci_set_node *current = ADCI_ALLOC(sizeof(struct adci_set_node));
    current->data = ADCI_ALLOC(set->bsize);
    memcpy(current->data, element, set->bsize);
    current->next = NULL;
    if(!adci_set_add_node(set, current)){
        ADCI_FREE(current->data);
        ADCI_FREE(current);
        return false;
    }
    return true;
}

bool adci_set_has(struct adci_set set, const void *element){
    unsigned int index = set.hasher(&set, element) % set.capacity;
    struct adci_set_node *current = set.data[index];
    while(current){
        if(memcmp(element, current->data, set.bsize) == 0) return true;
        current = current->next;
    }
    return false;
}

struct adci_set_iterator adci_set_get_iterator(struct adci_set *set){
    struct adci_set_iterator iterator = {.set = set, .done = false};
    for(unsigned int i = 0; i < set->capacity; i++){
        iterator.index = i;
        iterator.current = set->data[i];
        if(iterator.current) break;
    }
    return iterator;
}

void * adci_set_get_next(struct adci_set_iterator *iterator){
    if(iterator->done) return NULL;
    if(!iterator->current){
        for(++iterator->index; iterator->index < iterator->set->capacity; iterator->index++){
            if(!iterator->set->data[iterator->index]) continue;
            iterator->current = iterator->set->data[iterator->index];
            break;
        }
    }

    if(!iterator->current){
        iterator->done = true;
        return NULL;
    } 

    struct adci_set_node *current =  iterator->current;
    iterator->current = iterator->current->next;
    return current->data;
}