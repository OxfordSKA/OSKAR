/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "utility/oskar_getline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_getline(char** lineptr, size_t* n, FILE* stream)
{
    /* Initialise the byte counter. */
    size_t size = 0;
    int c = 0;

    /* Check if buffer is empty. */
    if (*n == 0 || *lineptr == 0)
    {
        *n = 1024;
        *lineptr = (char*)malloc(*n);
        if (*lineptr == 0) return OSKAR_ERR_MEMORY_ALLOC_FAILURE;
    }

    /* Read in the line. */
    for (;;)
    {
        /* Get the character. */
        c = getc(stream);

        /* Check if end-of-file or end-of-line has been reached. */
        if (c == EOF || c == '\n') break;

        /* Allocate space for size+1 bytes (including NULL terminator). */
        if (size + 1 >= *n)
        {
            void *t = 0;

            /* Double the length of the buffer. */
            *n = 2 * *n + 1;
            t = realloc(*lineptr, *n);
            if (!t) return OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            *lineptr = (char*)t;
        }

        /* Store the character. */
        (*lineptr)[size++] = c;
    }

    /* Add a NULL terminator. */
    (*lineptr)[size] = '\0';

    /* Return the number of characters read, or EOF as appropriate. */
    if (c == EOF && size == 0) return OSKAR_ERR_EOF;
    return (int) size;
}

#ifdef __cplusplus
}
#endif
