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

#include "utility/oskar_binary_stream_write.h"
#include "utility/oskar_mem_binary_stream_write.h"
#include "utility/oskar_mem_copy.h"
#include "utility/oskar_mem_element_size.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_init.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_mem_binary_stream_write(const oskar_Mem* mem, FILE* stream,
        unsigned char id_group, unsigned char id_tag, int user_index,
        int num_to_write)
{
    int err = 0, type, location, num_elements;
    oskar_Mem temp;
    size_t size_bytes;
    const oskar_Mem* data = NULL;

    /* Sanity check on inputs. */
    if (mem == NULL || stream == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Get the meta-data. */
    type = mem->type;
    location = mem->location;
    num_elements = mem->num_elements;

    /* Initialise temporary (to zero length). */
    oskar_mem_init(&temp, type, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE, &err);

    /* Get the total number of bytes to write. */
    if (num_to_write <= 0)
        num_to_write = num_elements;
    size_bytes = num_to_write * oskar_mem_element_size(type);

    /* Check if data is in CPU or GPU memory. */
    if (location == OSKAR_LOCATION_CPU)
    {
        data = mem;
    }
    else if (location == OSKAR_LOCATION_GPU)
    {
        /* Copy to temporary. */
        oskar_mem_copy(&temp, mem, &err);
        if (err)
        {
            oskar_mem_free(&temp, &err);
            return err;
        }
        data = &temp;
    }

    /* Save the memory to a binary stream. */
    err = oskar_binary_stream_write(stream, (unsigned char)type,
            id_group, id_tag, user_index, size_bytes, data->data);

    /* Free the temporary. */
    oskar_mem_free(&temp, &err);

    return err;
}

int oskar_mem_binary_stream_write_ext(const oskar_Mem* mem, FILE* stream,
        const char* name_group, const char* name_tag, int user_index,
        int num_to_write)
{
    int err = 0, type, location, num_elements;
    oskar_Mem temp;
    size_t size_bytes;
    const oskar_Mem* data = NULL;

    /* Sanity check on inputs. */
    if (mem == NULL || stream == NULL ||
            name_group == NULL || name_tag == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Get the meta-data. */
    type = mem->type;
    location = mem->location;
    num_elements = mem->num_elements;

    /* Initialise temporary (to zero length). */
    oskar_mem_init(&temp, type, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE, &err);

    /* Get the total number of bytes to write. */
    if (num_to_write <= 0)
        num_to_write = num_elements;
    size_bytes = num_to_write * oskar_mem_element_size(type);

    /* Check if data is in CPU or GPU memory. */
    if (location == OSKAR_LOCATION_CPU)
    {
        data = mem;
    }
    else if (location == OSKAR_LOCATION_GPU)
    {
        /* Copy to temporary. */
        oskar_mem_copy(&temp, mem, &err);
        if (err)
        {
            oskar_mem_free(&temp, &err);
            return err;
        }
        data = &temp;
    }

    /* Save the memory to a binary stream. */
    err = oskar_binary_stream_write_ext(stream, (unsigned char)type,
            name_group, name_tag, user_index, size_bytes, data->data);

    /* Free the temporary. */
    oskar_mem_free(&temp, &err);

    return err;
}

#ifdef __cplusplus
}
#endif
