/*
 * Copyright (c) 2011-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utility/oskar_string_to_array.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define strtok_r(s,d,p) strtok_s(s,d,p)
#else
/* Ensure function is declared for strange string.h header. */
/* NOLINTNEXTLINE(readability-identifier-naming) */
char* strtok_r(char*, const char*, char**);
#endif

#define DELIMITERS ", \t"

/* Integer. */
size_t oskar_string_to_array_i(const char* str, size_t n, int* data)
{
    size_t i = 0;
    char *end_ptr = 0;
    do
    {
        if (!str || '#' == *str || 0 == *str) break;
        const long int val = strtol(str, &end_ptr, 10);
        if (end_ptr > str)
        {
            data[i++] = (int) val;
            str = end_ptr;
        }
        else
        {
            str++;
        }
    } while (i < n);
    return i;
}


/* Double precision. */
size_t oskar_string_to_array_d(char* str, size_t n, double* data)
{
    size_t i = 0;
    char *save_ptr = 0, *token = 0, *end_ptr = 0;
    do
    {
        token = strtok_r(str, DELIMITERS, &save_ptr);
        str = NULL;
        if (!token || token[0] == '#') break;
        const double val = strtod(token, &end_ptr);
        if (end_ptr > token) data[i++] = val;
    }
    while (i < n);
    return i;
}

#ifdef __cplusplus
}
#endif
