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

#include "utility/oskar_mem_different.h"
#include "utility/oskar_mem_element_size.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_mem_different(const oskar_Mem* one, const oskar_Mem* two,
        int num_elements)
{
    int type, bytes_to_check, i;

    /* Sanity check on inputs. */
    if (one == NULL || two == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Check the data types. */
    type = one->type;
    if (type != two->type)
        return OSKAR_TRUE;

    /* Check the number of elements. */
    if (num_elements <= 0 || num_elements > one->num_elements)
        num_elements = one->num_elements;
    if (num_elements > two->num_elements)
        return OSKAR_TRUE;
    bytes_to_check = num_elements * oskar_mem_element_size(type);

    /* Check data location. */
    if (one->location == OSKAR_LOCATION_CPU &&
            two->location == OSKAR_LOCATION_CPU)
    {
        const char *p1, *p2;

        /* Check contents of CPU memory (every byte). */
        p1 = (const char*)(one->data);
        p2 = (const char*)(two->data);
        for (i = 0; i < bytes_to_check; ++i)
        {
            if (p1[i] != p2[i])
                return OSKAR_TRUE;
        }

        /* Memory contents must be the same. */
        return OSKAR_FALSE;
    }

    /* Data checks are currently only supported in CPU memory. */
    return OSKAR_ERR_BAD_LOCATION;
}

#ifdef __cplusplus
}
#endif
