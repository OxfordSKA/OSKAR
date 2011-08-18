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

#include "sky/oskar_load_sources.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_load_sources(const char* file_path, oskar_SkyModelGlobal_d* sky)
{
    // Open the file.
    FILE* file = fopen(file_path, "r");
    if (file == NULL) return;

    const double deg2rad = 0.0174532925199432957692;
    sky->num_sources = 0;
    sky->RA  = NULL;
    sky->Dec = NULL;
    sky->I   = NULL;
    sky->Q   = NULL;
    sky->U   = NULL;
    sky->V   = NULL;

    double ra, dec, I, Q, U, V;

    char line[1024];
    while (fgets(line, sizeof(line), file))
    {
        // Ignore comment lines (lines starting with '#')
        if (line[0] == '#')
            continue;

        // Load source co-ordinates.
        sscanf(line, "%lf %lf %lf %lf %lf %lf", &ra, &dec, &I, &Q, &U, &V);

        // Convert coordinates to radians.
        ra  *= deg2rad;
        dec *= deg2rad;

        // Ensure enough space in arrays.
        if (sky->num_sources % 100 == 0)
        {
            size_t mem_size = ((sky->num_sources) + 100) * sizeof(double);
            sky->RA  = (double*) realloc(sky->RA,  mem_size);
            sky->Dec = (double*) realloc(sky->Dec, mem_size);
            sky->I   = (double*) realloc(sky->I,   mem_size);
            sky->Q   = (double*) realloc(sky->Q,   mem_size);
            sky->U   = (double*) realloc(sky->U,   mem_size);
            sky->V   = (double*) realloc(sky->V,   mem_size);
        }

        sky->RA[sky->num_sources]  = ra;
        sky->Dec[sky->num_sources] = dec;
        sky->I[sky->num_sources]   = I;
        sky->Q[sky->num_sources]   = Q;
        sky->U[sky->num_sources]   = U;
        sky->V[sky->num_sources]   = V;
        sky->num_sources++;
    }
    fclose(file);
}

#ifdef __cplusplus
}
#endif
