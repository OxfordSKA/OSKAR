/*
 * Copyright (c) 2013-2015, The University of Oxford
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


#include "oskar_settings_load_tid_parameter_file.h"

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

static size_t string_to_array(char* str, size_t n, double* data);
static int oskar_settings_getline(char** lineptr, size_t* n, FILE* stream);

void oskar_settings_load_tid_parameter_file(oskar_SettingsTIDscreen* TID,
        const char* filename, int* status)
{
    char* line = NULL;
    size_t bufsize = 0;
    int n = 0;
    FILE* file;

    /* Initialise TID component arrays */
    TID->num_components = 0;
    TID->amp = NULL;
    TID->speed = NULL;
    TID->wavelength = NULL;
    TID->theta = NULL;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Open the file. */
    file = fopen(filename, "r");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Loop over each line in the file. */
    while (oskar_settings_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        size_t read = 0;

        /* Ignore comment lines (lines starting with '#'). */
        if (line[0] == '#') continue;

        /* Read the screen height */
        if (n == 0)
        {
            read = string_to_array(line, 1, &TID->height_km);
            if (read != 1) continue;
            ++n;
        }
        /* Read TID components */
        else
        {
            size_t newSize;
            double par[] = {0.0, 0.0, 0.0, 0.0};
            read = string_to_array(line, sizeof(par)/sizeof(double), par);
            if (read != 4) continue;

            /* Resize component arrays. */
            newSize = (TID->num_components+1) * sizeof(double);
            TID->amp = (double*)realloc(TID->amp, newSize);
            TID->speed = (double*)realloc(TID->speed, newSize);
            TID->wavelength = (double*)realloc(TID->wavelength, newSize);
            TID->theta = (double*)realloc(TID->theta, newSize);

            /* Store the component */
            TID->amp[TID->num_components] = par[0];
            TID->speed[TID->num_components] = par[1];
            TID->theta[TID->num_components] = par[2];
            TID->wavelength[TID->num_components] = par[3];
            ++n;
            ++(TID->num_components);
        }
    }

    /* Free the line buffer and close the file. */
    free(line);
    fclose(file);
}


size_t string_to_array(char* str, size_t n, double* data)
{
    size_t i = 0;
    char *save_ptr, *token;
    do
    {
        token = strtok_r(str, ", ", &save_ptr);
        str = NULL;
        if (!token) break;
        if (token[0] == '#') break;
        if (sscanf(token, "%lf", &data[i]) > 0) i++;
    }
    while (i < n);
    return i;
}


int oskar_settings_getline(char** lineptr, size_t* n, FILE* stream)
{
    /* Initialise the byte counter. */
    size_t size = 0;
    int c;

    /* Check if buffer is empty. */
    if (*n == 0 || *lineptr == 0)
    {
        *n = 80;
        *lineptr = (char*)malloc(*n);
        if (*lineptr == 0)
            return OSKAR_ERR_MEMORY_ALLOC_FAILURE;
    }

    /* Read in the line. */
    for (;;)
    {
        /* Get the character. */
        c = getc(stream);

        /* Check if end-of-file or end-of-line has been reached. */
        if (c == EOF || c == '\n')
            break;

        /* Allocate space for size+1 bytes (including NULL terminator). */
        if (size + 1 >= *n)
        {
            void *t;

            /* Double the length of the buffer. */
            *n = 2 * *n + 1;
            t = realloc(*lineptr, *n);
            if (!t)
                return OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            *lineptr = (char*)t;
        }

        /* Store the character. */
        (*lineptr)[size++] = c;
    }

    /* Add a NULL terminator. */
    (*lineptr)[size] = '\0';

    /* Return the number of characters read, or EOF as appropriate. */
    if (c == EOF && size == 0)
        return OSKAR_ERR_EOF;
    return size;
}


#ifdef __cplusplus
}
#endif
