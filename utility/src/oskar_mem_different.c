/*
 * Copyright (c) 2012-2015, The University of Oxford
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
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_mem_different(const oskar_Mem* one, const oskar_Mem* two,
        size_t num_elements, int* status)
{
    size_t bytes_to_check;

    /* Check if safe to proceed. */
    if (*status) return OSKAR_TRUE;

    /* Check that both arrays exist. */
    if ((!one && two) || (one && !two)) return OSKAR_TRUE;

    /* If neither array exists, return false. */
    if (!one && !two) return OSKAR_FALSE;

    /* Check the data types. */
    if (one->type != two->type) return OSKAR_TRUE;

    /* Check the number of elements. */
    if (num_elements == 0 || num_elements > one->num_elements)
        num_elements = one->num_elements;
    if (num_elements > two->num_elements)
        return OSKAR_TRUE;
    bytes_to_check = num_elements * oskar_mem_element_size(one->type);

    /* Check data location. */
    if (one->location == OSKAR_CPU && two->location == OSKAR_CPU)
        return (memcmp(one->data, two->data, bytes_to_check) != 0);

    /* Data checks are only supported in CPU memory. */
    *status = OSKAR_ERR_BAD_LOCATION;
    return OSKAR_TRUE;
}

#ifdef __cplusplus
}
#endif
