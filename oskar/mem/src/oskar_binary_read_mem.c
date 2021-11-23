/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_binary_read_mem.h"

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
    int type = 0;
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
    {
        oskar_mem_copy(mem, temp, status);
    }

    /* Free the temporary. */
    oskar_mem_free(temp, status);
}

void oskar_binary_read_mem_ext(oskar_Binary* handle, oskar_Mem* mem,
        const char* name_group, const char* name_tag, int user_index,
        int* status)
{
    int type = 0;
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
    {
        oskar_mem_copy(mem, temp, status);
    }

    /* Free the temporary. */
    oskar_mem_free(temp, status);
}

#ifdef __cplusplus
}
#endif
