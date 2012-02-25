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

#include "utility/oskar_mem_copy.h"
#include "utility/oskar_mem_element_size.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_load_binary.h"
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_binary_stream_read.h"
#include "utility/oskar_binary_tag_index_create.h"
#include "utility/oskar_binary_tag_index_query.h"

#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_mem_load_binary(oskar_Mem* mem, const char* filename,
        oskar_BinaryTagIndex** index, const char* name_group,
        const char* name_tag, int user_index)
{
    int err, type, location, num_elements, element_size;
    oskar_Mem temp;
    size_t size_bytes;
    oskar_Mem* data = NULL;
    FILE* stream;

    /* Sanity check on inputs. */
    if (mem == NULL || filename == NULL || index == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Get the meta-data. */
#ifdef __cplusplus
    type = mem->type();
    location = mem->location();
#else
    type = mem->private_type;
    location = mem->private_location;
#endif

    /* Initialise temporary (to zero length). */
    oskar_mem_init(&temp, type, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);

    /* Check if data is in CPU or GPU memory. */
    data = (location == OSKAR_LOCATION_CPU) ? mem : &temp;

    /* Open the input file. */
    stream = fopen(filename, "rb");
    if (stream == NULL)
        return OSKAR_ERR_FILE_IO;

    /* Create the tag index if it doesn't already exist. */
    if (*index == NULL)
    {
        err = oskar_binary_tag_index_create(index, stream);
        if (err)
        {
            fclose(stream);
            return err;
        }
    }

    /* Query the tag index to find out how big the block is. */
    element_size = oskar_mem_element_size(type);
    err = oskar_binary_tag_index_query(*index, (unsigned char)type, 0, 0,
            name_group, name_tag, user_index, NULL, &size_bytes, NULL);
    if (err)
    {
        fclose(stream);
        return err;
    }

    /* Resize memory block if necessary, so that it can hold the data. */
    num_elements = (int)ceil(size_bytes / element_size);
    err = oskar_mem_realloc(data, num_elements);
    if (err)
    {
        oskar_mem_free(&temp);
        fclose(stream);
        return err;
    }
    size_bytes = num_elements * element_size;

    /* Load the memory from a binary stream. */
    err = oskar_binary_stream_read(stream, index, (unsigned char)type,
            name_group, name_tag, user_index, size_bytes, data->data);

    /* Close the input file and check for errors. */
    fclose(stream);
    if (err)
    {
        oskar_mem_free(&temp);
        return err;
    }

    /* Copy to GPU memory if required. */
    if (location == OSKAR_LOCATION_GPU)
        err = oskar_mem_copy(mem, &temp);

    /* Free the temporary. */
    oskar_mem_free(&temp);

    return err;
}

#ifdef __cplusplus
}
#endif
