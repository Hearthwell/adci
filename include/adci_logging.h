#ifndef ADCI_LOGGING_H
#define ADCI_LOGGING_H

#include "adci_common.h"

/* TODO, ADD SUPPORT FOR THE LEVELS */
#define ADCI_LOG(_level, _fmt, ...) printf("I %s; " _fmt "\n", __func__, ##__VA_ARGS__)

enum adci_logging_level{
    ADCI_ERROR,
    ADCI_WARNING,
    ADCI_INFO,
    ADCI_DEBUG
};

#endif //ADCI_LOGGING_H