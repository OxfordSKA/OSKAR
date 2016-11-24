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

#include "mem/oskar_mem.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_random_range(oskar_Mem* mem, double lo, double hi, int* status)
{
    oskar_Mem *temp = 0, *ptr = 0;
    size_t i, num_elements;
    int location, precision, type;
    double r;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get meta-data and check type. */
    location = oskar_mem_location(mem);
    num_elements = oskar_mem_length(mem);
    type = oskar_mem_type(mem);
    precision = oskar_type_precision(type);
    if (precision != OSKAR_SINGLE && precision != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Initialise temporary memory array if required. */
    ptr = mem;
    if (location != OSKAR_CPU)
    {
        temp = oskar_mem_create(type, OSKAR_CPU, num_elements, status);
        if (*status)
        {
            oskar_mem_free(temp, status);
            return;
        }
        ptr = temp;
    }

    /* Get total number of elements. */
    if (oskar_type_is_matrix(type)) num_elements *= 4;
    if (oskar_type_is_complex(type)) num_elements *= 2;

    /* Fill memory with random numbers. */
    if (precision == OSKAR_SINGLE)
    {
        float *p;
        p = oskar_mem_float(ptr, status);
        for (i = 0; i < num_elements; ++i)
        {
            r = lo + (hi - lo) * rand() / (double)RAND_MAX;
            p[i] = (float)r;
        }
    }
    else if (precision == OSKAR_DOUBLE)
    {
        double *p;
        p = oskar_mem_double(ptr, status);
        for (i = 0; i < num_elements; ++i)
        {
            r = lo + (hi - lo) * rand() / (double)RAND_MAX;
            p[i] = r;
        }
    }

    /* Copy and clean up if required. */
    if (location != OSKAR_CPU)
        oskar_mem_copy(mem, ptr, status);
    oskar_mem_free(temp, status);
}

#ifdef __cplusplus
}
#endif
