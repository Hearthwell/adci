#ifndef ADCI_COMMON_H
#define ADCI_COMMON_H

#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>

/* PRESENT ON API FUNCTIONS, ANY FUNCTION WITHOUT THIS MACRO ON THE SIGNATURE IS NOT TO BE CALLED DIRECTLY */
#define ADCI_API

#ifdef ADCI_TEST
/* TO MAKE PRIVATE FUNCTIONS PUBLIC FOR TEST PURPOSES */
#define ADCI_TEST_VISIBILITY
#else
#define ADCI_TEST_VISIBILITY static
#endif

#define ADCI_ALLOC(_size) malloc(_size)
#define ADCI_FREE(_ptr) free(_ptr)

#endif //ADCI_COMMON_H