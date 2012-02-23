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
#include "utility/oskar_mem_save_binary_append.h"
#include "utility/oskar_binary_file_append.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_mem_save_binary_append(const oskar_Mem* mem, const char* filename,
        unsigned char id, unsigned char id_user_1, unsigned char id_user_2,
        int num_to_write)
{
    int err, type, location, num_elements;
    oskar_Mem temp;
    size_t size_bytes;
    const oskar_Mem* data = NULL;

    /* Sanity check on inputs. */
    if (mem == NULL || filename == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Get the meta-data. */
#ifdef __cplusplus
    type = mem->type();
    location = mem->location();
    num_elements = mem->num_elements();
#else
    type = mem->private_type;
    location = mem->private_location;
    num_elements = mem->private_num_elements;
#endif

    /* Initialise temporary (to zero length). */
    oskar_mem_init(&temp, type, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);

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
        err = oskar_mem_copy(&temp, mem);
        if (err)
        {
            oskar_mem_free(&temp);
            return err;
        }
        data = &temp;
    }

    /* Save the memory to a binary file. */
    err = oskar_binary_file_append(filename, id, id_user_1, id_user_2,
            type, size_bytes, data->data);

    /* Free the temporary. */
    oskar_mem_free(&temp);

    return err;
}

#ifdef __cplusplus
}
#endif
