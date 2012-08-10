/*
 * Copyright (c) 2012, The University of Oxford
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

#include "utility/oskar_mem_copy.h"
#include "utility/oskar_mem_get_pointer.h"
#include "utility/oskar_mem_insert.h"
#include "utility/oskar_mem_realloc.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_mem_copy(oskar_Mem* dst, const oskar_Mem* src)
{
    int error = 0;

    /* Sanity check on inputs. */
    if (src == NULL || dst == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Check the data types. */
    if (src->type != dst->type)
        return OSKAR_ERR_TYPE_MISMATCH;

    /* Only copy the pointer if destination does not own its memory. */
    if (dst->owner == OSKAR_FALSE)
    {
        /* FIXME is this really the behaviour we want? ... too confusing! */
        printf("\n*************\n");
        printf("************* If you see this message, please let us know!\n");
        printf("*************\n\n");
        fflush(stdout);

        /* Disallow a pointer copy at a different location. */
        if (dst->location != src->location)
            return OSKAR_ERR_BAD_LOCATION;

        oskar_mem_get_pointer(dst, src, 0, src->num_elements, &error);
        if (error) return error;
    }
    else
    {
        /* Check the data dimensions and resize if required. */
        if (src->num_elements > dst->num_elements)
        {
            error = oskar_mem_realloc(dst, src->num_elements);
            if (error) return error;
        }

        /* Copy the memory. */
        error = oskar_mem_insert(dst, src, 0);
        if (error) return error;
    }

    return error;
}

#ifdef __cplusplus
}
#endif
