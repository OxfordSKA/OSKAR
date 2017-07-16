/*
 * Copyright (c) 2016, The University of Oxford
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

#include "mem/oskar_mem.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_add_real(oskar_Mem* mem, double val, int* status)
{
    int precision, location;
    size_t i, num_elements;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get meta-data. */
    precision = oskar_mem_precision(mem);
    location = oskar_mem_location(mem);

    /* Check for empty array. */
    num_elements = oskar_mem_length(mem);
    if (num_elements == 0)
        return;

    /* Check location. */
    if (location != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Get total number of elements to add. */
    if (oskar_mem_is_matrix(mem))
        num_elements *= 4;

    /* Switch on type and location. */
    if (oskar_mem_is_complex(mem))
    {
        if (precision == OSKAR_DOUBLE)
        {
            double2 *t;
            t = oskar_mem_double2(mem, status);
            for (i = 0; i < num_elements; ++i) t[i].x += val;
        }
        else if (precision == OSKAR_SINGLE)
        {
            float2 *t;
            t = oskar_mem_float2(mem, status);
            for (i = 0; i < num_elements; ++i) t[i].x += val;
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else
    {
        if (precision == OSKAR_DOUBLE)
        {
            double *t;
            t = oskar_mem_double(mem, status);
            for (i = 0; i < num_elements; ++i) t[i] += val;
        }
        else if (precision == OSKAR_SINGLE)
        {
            float *t;
            t = oskar_mem_float(mem, status);
            for (i = 0; i < num_elements; ++i) t[i] += val;
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
}

#ifdef __cplusplus
}
#endif
