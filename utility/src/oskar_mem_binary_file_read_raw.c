/*
 * Copyright (c) 2012-2013, The University of Oxford
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
#include <oskar_mem_binary_file_read_raw.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_binary_file_read_raw(oskar_Mem* mem, const char* filename,
        int* status)
{
    int num_elements, element_size;
    oskar_Mem temp;
    size_t size_bytes;
    oskar_Mem* data = NULL;
    FILE* stream;

    /* Check all inputs. */
    if (!mem || !filename || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Initialise temporary (to zero length). */
    oskar_mem_init(&temp, mem->type, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE, status);

    /* Check if data is in CPU or GPU memory. */
    data = (mem->location == OSKAR_LOCATION_CPU) ? mem : &temp;

    /* Open the input file. */
    stream = fopen(filename, "rb");
    if (stream == NULL)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Get the file size. */
    fseek(stream, 0, SEEK_END);
    size_bytes = ftell(stream);

    /* Resize memory block so that it can hold the data. */
    element_size = oskar_mem_element_size(mem->type);
    num_elements = (int)ceil(size_bytes / element_size);
    oskar_mem_realloc(data, num_elements, status);
    if (*status)
    {
        oskar_mem_free(&temp, status);
        fclose(stream);
        return;
    }
    size_bytes = num_elements * element_size;

    /* Read the data. */
    fseek(stream, 0, SEEK_SET);
    if (fread(data->data, 1, size_bytes, stream) != size_bytes)
    {
        oskar_mem_free(&temp, status);
        fclose(stream);
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Close the input file. */
    fclose(stream);

    /* Copy to GPU memory if required. */
    if (mem->location == OSKAR_LOCATION_GPU)
        oskar_mem_copy(mem, &temp, status);

    /* Free the temporary. */
    oskar_mem_free(&temp, status);
}

#ifdef __cplusplus
}
#endif
