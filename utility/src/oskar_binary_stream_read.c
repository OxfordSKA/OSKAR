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

#include "utility/oskar_binary_stream_read.h"
#include "utility/oskar_binary_tag_index_create.h"
#include "utility/oskar_binary_tag_index_query.h"
#include "utility/oskar_Mem.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_binary_stream_read(FILE* stream, oskar_BinaryTagIndex** index,
        unsigned char data_type, unsigned char id_group, unsigned char id_tag,
        int user_index, size_t data_size, void* data, int* status)
{
    size_t block_size = 0;
    long block_offset = 0;

    /* Check all inputs. */
    if (!stream || !index || !data || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Index the stream if necessary. */
    if (*index == NULL)
    {
        oskar_binary_tag_index_create(index, stream, status);
        if (*status) return;
    }

    /* Query the tag index to get the block size and offset. */
    oskar_binary_tag_index_query(*index, data_type, id_group,
            id_tag, user_index, &block_size, &block_offset, status);
    if (*status) return;

    /* Check that there is enough memory in the block. */
    if (data_size < block_size)
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }

    /* Copy the data out of the stream. */
    if (fseek(stream, block_offset, SEEK_SET) != 0)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
    if (fread(data, 1, block_size, stream) != block_size)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
}

void oskar_binary_stream_read_double(FILE* stream,
        oskar_BinaryTagIndex** index, unsigned char id_group,
        unsigned char id_tag, int user_index, double* value, int* status)
{
    oskar_binary_stream_read(stream, index, OSKAR_DOUBLE,
            id_group, id_tag, user_index, sizeof(double), value, status);
}

void oskar_binary_stream_read_int(FILE* stream,
        oskar_BinaryTagIndex** index, unsigned char id_group,
        unsigned char id_tag, int user_index, int* value, int* status)
{
    oskar_binary_stream_read(stream, index, OSKAR_INT,
            id_group, id_tag, user_index, sizeof(int), value, status);
}

void oskar_binary_stream_read_ext(FILE* stream, oskar_BinaryTagIndex** index,
        unsigned char data_type, const char* name_group, const char* name_tag,
        int user_index, size_t data_size, void* data, int* status)
{
    size_t block_size = 0;
    long block_offset = 0;

    /* Check all inputs. */
    if (!stream || !index || !data || !name_group || !name_tag || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Index the stream if necessary. */
    if (*index == NULL)
    {
        oskar_binary_tag_index_create(index, stream, status);
        if (*status) return;
    }

    /* Query the tag index to get the block size and offset. */
    oskar_binary_tag_index_query_ext(*index, data_type, name_group,
            name_tag, user_index, NULL, &block_size, &block_offset, status);
    if (*status) return;

    /* Check that there is enough memory in the block. */
    if (data_size < block_size)
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }

    /* Copy the data out of the stream. */
    if (fseek(stream, block_offset, SEEK_SET) != 0)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
    if (fread(data, 1, block_size, stream) != block_size)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
}

void oskar_binary_stream_read_ext_double(FILE* stream,
        oskar_BinaryTagIndex** index,  const char* name_group,
        const char* name_tag, int user_index, double* value, int* status)
{
    oskar_binary_stream_read_ext(stream, index, OSKAR_DOUBLE,
            name_group, name_tag, user_index, sizeof(double), value, status);
}

void oskar_binary_stream_read_ext_int(FILE* stream,
        oskar_BinaryTagIndex** index, const char* name_group,
        const char* name_tag, int user_index, int* value, int* status)
{
    oskar_binary_stream_read_ext(stream, index, OSKAR_INT,
            name_group, name_tag, user_index, sizeof(int), value, status);
}

#ifdef __cplusplus
}
#endif
