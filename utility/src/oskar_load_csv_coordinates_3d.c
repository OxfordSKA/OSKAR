/*
 * Copyright (c) 2011, The University of Oxford
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

#include "utility/oskar_load_csv_coordinates_3d.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Single precision.
int oskar_load_csv_coordinates_3d_f(const char* filename, unsigned* n,
        float** x, float** y, float** z)
{
    // Open the file.
    FILE* file = fopen(filename, "r");
    if (file == NULL) return 0;
    *n = 0;
    *x = NULL;
    *y = NULL;
    *z = NULL;

    float ax, ay, az;
    while (fscanf(file, "%f,%f,%f", &ax, &ay, &az) == 3)
    {
        // Ensure enough space in arrays.
        if (*n % 100 == 0)
        {
            size_t mem_size = ((*n) + 100) * sizeof(float);
            *x = (float*) realloc(*x, mem_size);
            *y = (float*) realloc(*y, mem_size);
            *z = (float*) realloc(*z, mem_size);
        }

        // Store the data.
        (*x)[*n] = ax;
        (*y)[*n] = ay;
        (*z)[*n] = az;
        (*n)++;
    }
    fclose(file);

    return *n;
}

// Double precision.
int oskar_load_csv_coordinates_3d_d(const char* filename, unsigned* n,
        double** x, double** y, double** z)
{
    // Open the file.
    FILE* file = fopen(filename, "r");
    if (file == NULL) return 0;
    *n = 0;
    *x = NULL;
    *y = NULL;
    *z = NULL;

    double ax, ay, az;
    while (fscanf(file, "%lf,%lf,%lf", &ax, &ay, &az) == 3)
    {
        // Ensure enough space in arrays.
        if (*n % 100 == 0)
        {
            size_t mem_size = ((*n) + 100) * sizeof(double);
            *x = (double*) realloc(*x, mem_size);
            *y = (double*) realloc(*y, mem_size);
            *z = (double*) realloc(*z, mem_size);
        }

        // Store the data.
        (*x)[*n] = ax;
        (*y)[*n] = ay;
        (*z)[*n] = az;
        (*n)++;
    }
    fclose(file);

    return *n;
}

#ifdef __cplusplus
}
#endif
