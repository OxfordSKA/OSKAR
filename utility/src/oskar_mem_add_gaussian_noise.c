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


#include "utility/oskar_mem_add_gaussian_noise.h"
#include "utility/oskar_vector_types.h"
#include "math/oskar_random_gaussian.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_mem_add_gaussian_noise(oskar_Mem* mem, double stddev, double mean)
{
    int i;
    double r1, r2;

    if (mem == NULL) return OSKAR_ERR_INVALID_ARGUMENT;

    if (mem->location != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    if (mem->type == OSKAR_DOUBLE)
    {
        for (i = 0; i < mem->num_elements; ++i)
        {
            r1 = oskar_random_gaussian(NULL);
            ((double*)mem->data)[i] += r1 * stddev + mean;
        }
    }
    else if (mem->type == OSKAR_DOUBLE_COMPLEX)
    {
        for (i = 0; i < mem->num_elements; ++i)
        {
            r1 = oskar_random_gaussian(&r2);
            ((double2*)mem->data)[i].x += r1 * stddev + mean;
            ((double2*)mem->data)[i].y += r2 * stddev + mean;
        }
    }
    else if (mem->type == OSKAR_DOUBLE_COMPLEX_MATRIX)
    {
        for (i = 0; i < mem->num_elements; ++i)
        {
            r1 = oskar_random_gaussian(&r2);
            ((double4c*)mem->data)[i].a.x += r1 * stddev + mean;
            ((double4c*)mem->data)[i].a.y += r2 * stddev + mean;
            r1 = oskar_random_gaussian(&r2);
            ((double4c*)mem->data)[i].b.x += r1 * stddev + mean;
            ((double4c*)mem->data)[i].b.y += r2 * stddev + mean;
            r1 = oskar_random_gaussian(&r2);
            ((double4c*)mem->data)[i].c.x += r1 * stddev + mean;
            ((double4c*)mem->data)[i].c.y += r2 * stddev + mean;
            r1 = oskar_random_gaussian(&r2);
            ((double4c*)mem->data)[i].d.x += r1 * stddev + mean;
            ((double4c*)mem->data)[i].d.y += r2 * stddev + mean;
        }
    }
    else
        return OSKAR_ERR_BAD_DATA_TYPE;

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
