/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_binary_write_mem.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_binary_write_mem(oskar_Binary* handle, const oskar_Mem* mem,
        unsigned char id_group, unsigned char id_tag, int user_index,
        size_t num_to_write, int* status)
{
    int type = 0;
    oskar_Mem *temp = 0;
    size_t size_bytes = 0;
    const oskar_Mem* data = 0;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the data type. */
    type = oskar_mem_type(mem);

    /* Get the total number of bytes to write. */
    if (num_to_write == 0)
    {
        num_to_write = oskar_mem_length(mem);
    }
    size_bytes = num_to_write * oskar_mem_element_size(type);

    /* Check if data is in CPU or GPU memory. */
    data = mem;
    if (oskar_mem_location(mem) != OSKAR_CPU)
    {
        /* Copy to temporary. */
        temp = oskar_mem_create_copy(mem, OSKAR_CPU, status);
        data = temp;
    }

    /* Save the memory to a binary stream. */
    oskar_binary_write(handle, (unsigned char)type,
            id_group, id_tag, user_index, size_bytes,
            oskar_mem_void_const(data), status);

    /* Free the temporary. */
    oskar_mem_free(temp, status);
}

void oskar_binary_write_mem_ext(oskar_Binary* handle, const oskar_Mem* mem,
        const char* name_group, const char* name_tag, int user_index,
        size_t num_to_write, int* status)
{
    int type = 0;
    oskar_Mem *temp = 0;
    size_t size_bytes = 0;
    const oskar_Mem* data = 0;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the data type. */
    type = oskar_mem_type(mem);

    /* Get the total number of bytes to write. */
    if (num_to_write == 0)
    {
        num_to_write = oskar_mem_length(mem);
    }
    size_bytes = num_to_write * oskar_mem_element_size(type);

    /* Check if data is in CPU or GPU memory. */
    data = mem;
    if (oskar_mem_location(mem) != OSKAR_CPU)
    {
        /* Copy to temporary. */
        temp = oskar_mem_create_copy(mem, OSKAR_CPU, status);
        data = temp;
    }

    /* Save the memory to a binary stream. */
    oskar_binary_write_ext(handle, (unsigned char)type,
            name_group, name_tag, user_index, size_bytes,
            oskar_mem_void_const(data), status);

    /* Free the temporary. */
    oskar_mem_free(temp, status);
}

#ifdef __cplusplus
}
#endif
