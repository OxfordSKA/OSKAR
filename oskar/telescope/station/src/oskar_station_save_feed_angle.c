/*
 * Copyright (c) 2015, The University of Oxford
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

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"
#include "math/oskar_cmath.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define R2D (180.0 / M_PI)

void oskar_station_save_feed_angle(const char* filename,
        const oskar_Station* station, int x_pol, int* status)
{
    int i, num_elements;
    FILE* file;
    const double *a, *b, *c;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get pointers to the arrays. */
    if (x_pol)
    {
        a = oskar_mem_double_const(station->element_x_alpha_cpu, status);
        b = oskar_mem_double_const(station->element_x_beta_cpu, status);
        c = oskar_mem_double_const(station->element_x_gamma_cpu, status);
    }
    else
    {
        a = oskar_mem_double_const(station->element_y_alpha_cpu, status);
        b = oskar_mem_double_const(station->element_y_beta_cpu, status);
        c = oskar_mem_double_const(station->element_y_gamma_cpu, status);
    }

    /* Open the file. */
    file = fopen(filename, "w");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Save the data. */
    num_elements = oskar_station_num_elements(station);
    for (i = 0; i < num_elements; ++i)
    {
        fprintf(file, "% 14.6f % 14.6f % 14.6f\n",
                a[i] * R2D, b[i] * R2D, c[i] * R2D);
    }

    /* Close the file. */
    fclose(file);
}

#ifdef __cplusplus
}
#endif
