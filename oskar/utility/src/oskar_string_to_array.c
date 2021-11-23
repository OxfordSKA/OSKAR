/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "utility/oskar_string_to_array.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define strtok_r(s,d,p) strtok_s(s,d,p)
#else
/* Ensure function is declared for strange string.h header. */
char* strtok_r(char*, const char*, char**);
#endif

#define DELIMITERS ", \t"

/* Integer. */
size_t oskar_string_to_array_i(char* str, size_t n, int* data)
{
    size_t i = 0;
    char *save_ptr = 0, *token = 0;
    do
    {
        token = strtok_r(str, DELIMITERS, &save_ptr);
        str = NULL;
        if (!token || token[0] == '#') break;
        if (sscanf(token, "%i", &data[i]) > 0) i++;
    }
    while (i < n);
    return i;
}

/* Single precision. */
size_t oskar_string_to_array_f(char* str, size_t n, float* data)
{
    size_t i = 0;
    char *save_ptr = 0, *token = 0;
    do
    {
        token = strtok_r(str, DELIMITERS, &save_ptr);
        str = NULL;
        if (!token || token[0] == '#') break;
        if (sscanf(token, "%f", &data[i]) > 0) i++;
    }
    while (i < n);
    return i;
}

/* Double precision. */
size_t oskar_string_to_array_d(char* str, size_t n, double* data)
{
    size_t i = 0;
    char *save_ptr = 0, *token = 0;
    do
    {
        token = strtok_r(str, DELIMITERS, &save_ptr);
        str = NULL;
        if (!token || token[0] == '#') break;
        if (sscanf(token, "%lf", &data[i]) > 0) i++;
    }
    while (i < n);
    return i;
}

/* String array. */
size_t oskar_string_to_array_s(char* str, size_t n, char** data)
{
    size_t i = 0;
    char *save_ptr = 0;
    for (i = 0; i < n; ++i)
    {
        data[i] = strtok_r(str, DELIMITERS, &save_ptr);
        str = NULL;
        if (!data[i]) break;
        if (data[i][0] == '#') break;
    }
    return i;
}

/* Double precision. */
size_t oskar_string_to_array_realloc_d(char* str, size_t* n, double** data)
{
    size_t i = 0;
    char *save_ptr = 0, *token = 0;
    for (;;)
    {
        double val = 0.0;
        token = strtok_r(str, DELIMITERS, &save_ptr);
        str = NULL;
        if (!token || token[0] == '#') break;
        if (sscanf(token, "%lf", &val) > 0)
        {
            i++;
            if (*n < i || !(*data))
            {
                *n += 20;
                void* t = realloc(*data, (*n) * sizeof(double));
                if (!t) break;
                *data = (double*) t;
            }
            (*data)[i - 1] = val;
        }
    }
    return i;
}

/* String array. */
size_t oskar_string_to_array_realloc_s(char* str, size_t* n, char*** data)
{
    size_t i = 0;
    char *save_ptr = 0, *token = 0;
    for (i = 0; ; ++i)
    {
        token = strtok_r(str, DELIMITERS, &save_ptr);
        str = NULL;
        if (!token || token[0] == '#') break;

        /* Ensure array is big enough. */
        if (*n <= i || !(*data))
        {
            void* t = realloc(*data, ((*n) + 1) * sizeof(char*));
            if (!t) break;
            *data = (char**) t;
            ++(*n);
        }
        (*data)[i] = token;
    }
    return i;
}

#ifdef __cplusplus
}
#endif
