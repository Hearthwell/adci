#include "adci_common.h"

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