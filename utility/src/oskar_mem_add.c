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

#include <private_mem.h>
#include <oskar_mem.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* a = b + c */
void oskar_mem_add(oskar_Mem* a, const oskar_Mem* b, const oskar_Mem* c,
        int* status)
{
    size_t i, num_elements;

    /* Check all inputs. */
    if (!a || !b || !c || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check data types. */
    if (b->type != c->type ||
            a->type != b->type ||
            a->type != c->type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
    }

    /* Check number of elements. */
    if (b->num_elements != c->num_elements ||
            a->num_elements != b->num_elements ||
            a->num_elements != c->num_elements)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
    }

    /* Check locations. */
    if (b->location != c->location ||
            a->location != b->location ||
            a->location != c->location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
    }

    /* Note that device memory is not supported. */
    if (a->location == OSKAR_GPU)
        *status = OSKAR_ERR_BAD_LOCATION;

    if (a->data == NULL || b->data == NULL || c->data == NULL)
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    /* Get the total number of elements. */
    num_elements = a->num_elements;
    if (oskar_mem_type_is_matrix(a->type))
        num_elements *= 4;
    if (oskar_mem_type_is_complex(a->type))
        num_elements *= 2;

    /* Check if safe to proceed. */
    if (*status) return;

    if (oskar_mem_type_is_double(a->type))
    {
        double *aa, *bb, *cc;
        aa = (double*)a->data;
        bb = (double*)b->data;
        cc = (double*)c->data;
        for (i = 0; i < num_elements; ++i) aa[i] = bb[i] + cc[i];
    }
    else if (oskar_mem_type_is_single(a->type))
    {
        float *aa, *bb, *cc;
        aa = (float*)a->data;
        bb = (float*)b->data;
        cc = (float*)c->data;
        for (i = 0; i < num_elements; ++i) aa[i] = bb[i] + cc[i];
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
}

#ifdef __cplusplus
}
#endif
