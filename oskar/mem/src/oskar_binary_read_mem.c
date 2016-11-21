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

#include <oskar_binary_read_mem.h>

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_binary_read_mem(oskar_Binary* handle, oskar_Mem* mem,
        unsigned char id_group, unsigned char id_tag, int user_index,
        int* status)
{
    int type;
    oskar_Mem *temp = 0, *data = 0;
    size_t size_bytes = 0, element_size = 0;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the data type. */
    type = oskar_mem_type(mem);

    /* Initialise temporary (to zero length). */
    temp = oskar_mem_create(type, OSKAR_CPU, 0, status);

    /* Check if data is in CPU or GPU memory. */
    data = (oskar_mem_location(mem) == OSKAR_CPU) ? mem : temp;

    /* Query the tag index to find out how big the block is. */
    element_size = oskar_mem_element_size(type);
    oskar_binary_query(handle, (unsigned char)type,
            id_group, id_tag, user_index, &size_bytes, status);

    /* Resize memory block if necessary, so that it can hold the data. */
    oskar_mem_realloc(data, size_bytes / element_size, status);

    /* Load the memory. */
    oskar_binary_read(handle, (unsigned char)type, id_group, id_tag,
            user_index, size_bytes, oskar_mem_void(data), status);

    /* Copy to GPU memory if required. */
    if (oskar_mem_location(mem) != OSKAR_CPU)
        oskar_mem_copy(mem, temp, status);

    /* Free the temporary. */
    oskar_mem_free(temp, status);
}

void oskar_binary_read_mem_ext(oskar_Binary* handle, oskar_Mem* mem,
        const char* name_group, const char* name_tag, int user_index,
        int* status)
{
    int type;
    oskar_Mem *temp = 0, *data = 0;
    size_t size_bytes = 0, element_size = 0;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the data type. */
    type = oskar_mem_type(mem);

    /* Initialise temporary (to zero length). */
    temp = oskar_mem_create(type, OSKAR_CPU, 0, status);

    /* Check if data is in CPU or GPU memory. */
    data = (oskar_mem_location(mem) == OSKAR_CPU) ? mem : temp;

    /* Query the tag index to find out how big the block is. */
    element_size = oskar_mem_element_size(type);
    oskar_binary_query_ext(handle, (unsigned char)type,
            name_group, name_tag, user_index, &size_bytes, status);

    /* Resize memory block if necessary, so that it can hold the data. */
    oskar_mem_realloc(data, size_bytes / element_size, status);

    /* Load the memory. */
    oskar_binary_read_ext(handle, (unsigned char)type, name_group, name_tag,
            user_index, size_bytes, oskar_mem_void(data), status);

    /* Copy to GPU memory if required. */
    if (oskar_mem_location(mem) != OSKAR_CPU)
        oskar_mem_copy(mem, temp, status);

    /* Free the temporary. */
    oskar_mem_free(temp, status);
}

#ifdef __cplusplus
}
#endif
