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

#include "utility/oskar_BinaryTag.h"
#include "utility/oskar_binary_stream_write.h"
#include "utility/oskar_endian.h"
#include <oskar_mem.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_binary_stream_write(FILE* stream, unsigned char data_type,
        unsigned char id_group, unsigned char id_tag, int user_index,
        size_t data_size, const void* data, int* status)
{
    oskar_BinaryTag tag;
    size_t block_size;

    /* Initialise the tag. */
    char magic[] = "TAG";
    strcpy(tag.magic, magic);
    memset(tag.size_bytes, 0, sizeof(tag.size_bytes));
    memset(tag.user_index, 0, sizeof(tag.user_index));

    /* Check all inputs. */
    if (!stream || !data || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Set up the tag identifiers */
    tag.flags = 0;
    tag.data_type = data_type;
    tag.group.id = id_group;
    tag.tag.id = id_tag;

    /* Get the number of bytes in the block in little-endian byte order. */
    block_size = data_size;
    if (sizeof(size_t) != 4 && sizeof(size_t) != 8)
    {
        *status = OSKAR_ERR_BAD_BINARY_FORMAT;
        return;
    }
    if (oskar_endian() != OSKAR_LITTLE_ENDIAN)
    {
        if (sizeof(size_t) == 4)
            oskar_endian_swap_4((char*)&block_size);
        else if (sizeof(size_t) == 8)
            oskar_endian_swap_8((char*)&block_size);
    }

    /* Copy the block size in bytes to the tag, as little endian. */
    memcpy(tag.size_bytes, &block_size, sizeof(size_t));

    /* Get the user index in little-endian byte order. */
    if (oskar_endian() != OSKAR_LITTLE_ENDIAN)
    {
        if (sizeof(int) == 2)
            oskar_endian_swap_2((char*)&user_index);
        else if (sizeof(int) == 4)
            oskar_endian_swap_4((char*)&user_index);
        else if (sizeof(int) == 8)
            oskar_endian_swap_8((char*)&user_index);
    }

    /* Copy the user index to the tag, as little endian. */
    memcpy(tag.user_index, &user_index, sizeof(int));

    /* Write the tag to the file. */
    if (fwrite(&tag, sizeof(oskar_BinaryTag), 1, stream) != 1)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Write the data to the file. */
    if (fwrite(data, 1, data_size, stream) != data_size)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
}

void oskar_binary_stream_write_double(FILE* stream, unsigned char id_group,
        unsigned char id_tag, int user_index, double value, int* status)
{
    oskar_binary_stream_write(stream, OSKAR_DOUBLE, id_group, id_tag,
            user_index, sizeof(double), &value, status);
}

void oskar_binary_stream_write_int(FILE* stream, unsigned char id_group,
        unsigned char id_tag, int user_index, int value, int* status)
{
    oskar_binary_stream_write(stream, OSKAR_INT, id_group, id_tag,
            user_index, sizeof(int), &value, status);
}

void oskar_binary_stream_write_ext(FILE* stream, unsigned char data_type,
        const char* name_group, const char* name_tag, int user_index,
        size_t data_size, const void* data, int* status)
{
    oskar_BinaryTag tag;
    size_t block_size, lgroup, ltag;

    /* Initialise the tag. */
    char magic[] = "TAG";
    strcpy(tag.magic, magic);
    memset(tag.size_bytes, 0, sizeof(tag.size_bytes));
    memset(tag.user_index, 0, sizeof(tag.user_index));

    /* Check all inputs. */
    if (!stream || !data || !name_group || !name_tag || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check that string lengths are within range. */
    lgroup = strlen(name_group);
    ltag = strlen(name_tag);
    if (lgroup > 254 || ltag > 254)
    {
        *status = OSKAR_ERR_BINARY_TAG_TOO_LONG;
        return;
    }

    /* Set up the tag identifiers */
    tag.flags = 0;
    tag.flags |= (1 << 7); /* Set bit 7 to indicate that tag is extended. */
    tag.data_type = data_type;
    tag.group.bytes = 1 + (unsigned char)lgroup;
    tag.tag.bytes = 1 + (unsigned char)ltag;

    /* Get the number of bytes in the block in little-endian byte order. */
    block_size = data_size + tag.group.bytes + tag.tag.bytes;
    if (sizeof(size_t) != 4 && sizeof(size_t) != 8)
    {
        *status = OSKAR_ERR_BAD_BINARY_FORMAT;
        return;
    }
    if (oskar_endian() != OSKAR_LITTLE_ENDIAN)
    {
        if (sizeof(size_t) == 4)
            oskar_endian_swap_4((char*)&block_size);
        else if (sizeof(size_t) == 8)
            oskar_endian_swap_8((char*)&block_size);
    }

    /* Copy the block size in bytes to the tag, as little endian. */
    memcpy(tag.size_bytes, &block_size, sizeof(size_t));

    /* Get the user index in little-endian byte order. */
    if (oskar_endian() != OSKAR_LITTLE_ENDIAN)
    {
        if (sizeof(int) == 2)
            oskar_endian_swap_2((char*)&user_index);
        else if (sizeof(int) == 4)
            oskar_endian_swap_4((char*)&user_index);
        else if (sizeof(int) == 8)
            oskar_endian_swap_8((char*)&user_index);
    }

    /* Copy the user index to the tag, as little endian. */
    memcpy(tag.user_index, &user_index, sizeof(int));

    /* Write the tag to the file. */
    if (fwrite(&tag, sizeof(oskar_BinaryTag), 1, stream) != 1)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Write the group name and tag name to the file. */
    if (fwrite(name_group, 1, tag.group.bytes, stream) != tag.group.bytes)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
    if (fwrite(name_tag, 1, tag.tag.bytes, stream) != tag.tag.bytes)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Write the data to the file. */
    if (fwrite(data, 1, data_size, stream) != data_size)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
}

void oskar_binary_stream_write_ext_double(FILE* stream, const char* name_group,
        const char* name_tag, int user_index, double value, int* status)
{
    oskar_binary_stream_write_ext(stream, OSKAR_DOUBLE, name_group,
            name_tag, user_index, sizeof(double), &value, status);
}

void oskar_binary_stream_write_ext_int(FILE* stream, const char* name_group,
        const char* name_tag, int user_index, int value, int* status)
{
    oskar_binary_stream_write_ext(stream, OSKAR_INT, name_group,
            name_tag, user_index, sizeof(int), &value, status);
}

#ifdef __cplusplus
}
#endif
