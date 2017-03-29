/*
 * Copyright (c) 2012-2017, The University of Oxford
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

#include "binary/oskar_binary.h"
#include "binary/private_binary.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_binary_read_block(oskar_Binary* handle,
        int chunk_index, size_t data_size, void* data, int* status)
{
    size_t bytes = 0, chunk_size = 1 << 29;
    char* p;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check file was opened for reading. */
    if (handle->open_mode != 'r')
    {
        *status = OSKAR_ERR_BINARY_NOT_OPEN_FOR_READ;
        return;
    }

    /* Check index is in range. */
    if (chunk_index < 0 || chunk_index >= handle->num_chunks)
    {
        *status = OSKAR_ERR_BINARY_TAG_OUT_OF_RANGE;
        return;
    }

    /* Return if no data to read. */
    if (handle->payload_size_bytes[chunk_index] == 0) return;

    /* Check that there is enough memory in the block. */
    if (!data || data_size < handle->payload_size_bytes[chunk_index])
    {
        *status = OSKAR_ERR_BINARY_MEMORY_NOT_ALLOCATED;
        return;
    }

    /* Copy the data out of the stream. */
    if (fseek(handle->stream,
            handle->payload_offset_bytes[chunk_index], SEEK_SET) != 0)
    {
        *status = OSKAR_ERR_BINARY_SEEK_FAIL;
        return;
    }

    /* Read the data in chunks of 2^29 bytes (512 MB). */
    /* This works around a bug in some versions of fread() which are
     * limited to reading a maximum of 2 GB at once. */
    for (p = (char*)data, bytes = handle->payload_size_bytes[chunk_index];
            bytes > 0; p += chunk_size)
    {
        if (bytes < chunk_size) chunk_size = bytes;
        if (fread(p, 1, chunk_size, handle->stream) != chunk_size)
        {
            *status = OSKAR_ERR_BINARY_READ_FAIL;
            return;
        }
        bytes -= chunk_size;
    }

    /* Check CRC-32 code, if present. */
    if (handle->crc[chunk_index])
    {
        unsigned long crc;
        crc = handle->crc_header[chunk_index];
        crc = oskar_crc_update(handle->crc_data, crc, data,
                handle->payload_size_bytes[chunk_index]);
        if (crc != handle->crc[chunk_index])
            *status = OSKAR_ERR_BINARY_CRC_FAIL;
    }
}

void oskar_binary_read(oskar_Binary* handle,
        unsigned char data_type, unsigned char id_group, unsigned char id_tag,
        int user_index, size_t data_size, void* data, int* status)
{
    int chunk_index;
    if (*status) return;

    /* Check file was opened for reading. */
    if (handle->open_mode != 'r')
    {
        *status = OSKAR_ERR_BINARY_NOT_OPEN_FOR_READ;
        return;
    }

    /* Query the tag index to get the block size and offset. */
    chunk_index = oskar_binary_query(handle, data_type,
            id_group, id_tag, user_index, 0, status);
    oskar_binary_read_block(handle, chunk_index, data_size, data, status);
}

void oskar_binary_read_double(oskar_Binary* handle, unsigned char id_group,
        unsigned char id_tag, int user_index, double* value, int* status)
{
    oskar_binary_read(handle, OSKAR_DOUBLE,
            id_group, id_tag, user_index, sizeof(double), value, status);
}

void oskar_binary_read_int(oskar_Binary* handle, unsigned char id_group,
        unsigned char id_tag, int user_index, int* value, int* status)
{
    oskar_binary_read(handle, OSKAR_INT,
            id_group, id_tag, user_index, sizeof(int), value, status);
}

void oskar_binary_read_ext(oskar_Binary* handle,
        unsigned char data_type, const char* name_group, const char* name_tag,
        int user_index, size_t data_size, void* data, int* status)
{
    int chunk_index;
    if (*status) return;

    /* Check file was opened for reading. */
    if (handle->open_mode != 'r')
    {
        *status = OSKAR_ERR_BINARY_NOT_OPEN_FOR_READ;
        return;
    }

    /* Query the tag index to get the block size and offset. */
    chunk_index = oskar_binary_query_ext(handle, data_type,
            name_group, name_tag, user_index, 0, status);
    oskar_binary_read_block(handle, chunk_index, data_size, data, status);
}

void oskar_binary_read_ext_double(oskar_Binary* handle, const char* name_group,
        const char* name_tag, int user_index, double* value, int* status)
{
    oskar_binary_read_ext(handle, OSKAR_DOUBLE,
            name_group, name_tag, user_index, sizeof(double), value, status);
}

void oskar_binary_read_ext_int(oskar_Binary* handle, const char* name_group,
        const char* name_tag, int user_index, int* value, int* status)
{
    oskar_binary_read_ext(handle, OSKAR_INT,
            name_group, name_tag, user_index, sizeof(int), value, status);
}

#ifdef __cplusplus
}
#endif
