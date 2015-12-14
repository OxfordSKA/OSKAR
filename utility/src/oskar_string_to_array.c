/*
 * Copyright (c) 2011-2014, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <oskar_string_to_array.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define strtok_r(s,d,p) strtok_s(s,d,p)
#endif

#define DELIMITERS ", "

/* Integer. */
size_t oskar_string_to_array_i(char* str, size_t n, int* data)
{
    size_t i = 0;
    char *save_ptr, *token;
    do
    {
        token = strtok_r(str, DELIMITERS, &save_ptr);
        str = NULL;
        if (!token) break;
        if (token[0] == '#') break;
        if (sscanf(token, "%i", &data[i]) > 0) i++;
    }
    while (i < n);
    return i;
}

/* Single precision. */
size_t oskar_string_to_array_f(char* str, size_t n, float* data)
{
    size_t i = 0;
    char *save_ptr, *token;
    do
    {
        token = strtok_r(str, DELIMITERS, &save_ptr);
        str = NULL;
        if (!token) break;
        if (token[0] == '#') break;
        if (sscanf(token, "%f", &data[i]) > 0) i++;
    }
    while (i < n);
    return i;
}

/* Double precision. */
size_t oskar_string_to_array_d(char* str, size_t n, double* data)
{
    size_t i = 0;
    char *save_ptr, *token;
    do
    {
        token = strtok_r(str, DELIMITERS, &save_ptr);
        str = NULL;
        if (!token) break;
        if (token[0] == '#') break;
        if (sscanf(token, "%lf", &data[i]) > 0) i++;
    }
    while (i < n);
    return i;
}

/* String array. */
size_t oskar_string_to_array_s(char* str, size_t n, char** data)
{
    size_t i;
    char *save_ptr;
    for (i = 0; i < n; ++i)
    {
        data[i] = strtok_r(str, DELIMITERS, &save_ptr);
        str = NULL;
        if (!data[i]) break;
        if (data[i][0] == '#') break;
    }
    return i;
}

/* String array. */
size_t oskar_string_to_array_realloc_s(char* str, size_t* n, char*** data)
{
    size_t i;
    char *save_ptr, *token;
    for (i = 0; ; ++i)
    {
        token = strtok_r(str, DELIMITERS, &save_ptr);
        str = NULL;
        if (!token) break;
        if (token[0] == '#') break;

        /* Ensure array is big enough. */
        if (*n <= i || !(*data))
        {
            void* t;
            t = realloc(*data, ((*n) + 1) * sizeof(char*));
            if (!t) break;
            *data = t;
            ++(*n);
        }
        (*data)[i] = token;
    }
    return i;
}

#ifdef __cplusplus
}
#endif

