#ifndef ADCI_COMMON_H
#define ADCI_COMMON_H

#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>

/* PRESENT ON API FUNCTIONS, ANY FUNCTION WITHOUT THIS MACRO ON THE SIGNATURE IS NOT TO BE CALLED DIRECTLY */
#define ADCI_API

/* PRESENT ON FUNCTIONS WITH ASSERTIONS THAT CAN KILL THE PROGRAM */
#define ADCI_EXIT_POINT

#ifdef ADCI_TEST
/* TO MAKE PRIVATE FUNCTIONS PUBLIC FOR TEST PURPOSES */
#define ADCI_TEST_VISIBILITY
#else
#define ADCI_TEST_VISIBILITY static
#endif

#define ADCI_ALLOC(_size) malloc(_size)
#define ADCI_REALLOC(_ptr, _size) realloc(_ptr, _size)
#define ADCI_FREE(_ptr) free(_ptr)

struct adci_string{
    char *str;
    unsigned int size;
};

struct adci_string * adci_init_str(const char *str, unsigned int length);
bool adci_clean_str(struct adci_string *str);

struct adci_vector{
    void *data;
    unsigned int length;
    unsigned int capacity;
    unsigned int bsize;
};

struct adci_vector adci_vector_init(unsigned int element_bsize);
struct adci_vector adci_vector_from_array(void *elements, unsigned int count, unsigned int element_bsize);
bool adci_vector_add(struct adci_vector *vector, const void *element);
bool adci_vector_remove(struct adci_vector *vector, const void *element);
void * adci_vector_get(const struct adci_vector *vector, unsigned int index);
bool adci_vector_free(struct adci_vector *vector);

#endif //ADCI_COMMON_H