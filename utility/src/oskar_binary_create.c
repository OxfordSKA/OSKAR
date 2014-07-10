/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#include <private_binary.h>
#include <oskar_binary_create.h>
#include <oskar_endian.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

static void oskar_binary_resize(oskar_Binary* handle, int m);
static void oskar_binary_read_header(FILE* stream, oskar_BinaryHeader* header,
        int* status);
static void oskar_binary_write_header(FILE* stream, int* status);


oskar_Binary* oskar_binary_create(const char* filename, char mode, int* status)
{
    oskar_Binary* handle;
    oskar_BinaryHeader header;
    FILE* stream;
    int i;

    /* Check all inputs. */
    if (!filename || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Open the file and check or write the header, depending on the mode. */
    if (mode == 'r')
    {
        stream = fopen(filename, "rb");
        if (!stream)
        {
            *status = OSKAR_ERR_FILE_IO;
            return 0;
        }
        oskar_binary_read_header(stream, &header, status);
        if (*status) return 0;
    }
    else if (mode == 'w')
    {
        stream = fopen(filename, "wb");
        if (!stream)
        {
            *status = OSKAR_ERR_FILE_IO;
            return 0;
        }
        oskar_binary_write_header(stream, status);
    }
    else if (mode == 'a')
    {
        stream = fopen(filename, "a+b");
        if (!stream)
        {
            *status = OSKAR_ERR_FILE_IO;
            return 0;
        }

        /* Write header only if the file is empty. */
        fseek(stream, 0, SEEK_END);
        if (ftell(stream) == 0)
            oskar_binary_write_header(stream, status);
    }
    else
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Allocate index and store the stream handle. */
    handle = (oskar_Binary*) malloc(sizeof(oskar_Binary));
    handle->stream = stream;

    /* Initialise tag index. */
    handle->num_tags = 0;
    handle->extended = 0;
    handle->data_type = 0;
    handle->id_group = 0;
    handle->id_tag = 0;
    handle->name_group = 0;
    handle->name_tag = 0;
    handle->user_index = 0;
    handle->data_offset_bytes = 0;
    handle->data_size_bytes = 0;
    handle->block_size_bytes = 0;

    /* Finish if writing. */
    if (mode == 'w')
        return handle;

    /* Store the contents of the header for later use. */
    handle->bin_version     = header.bin_version;
    handle->big_endian      = header.endian;
    handle->size_ptr        = header.size_ptr;
    handle->oskar_ver_major = header.version[2];
    handle->oskar_ver_minor = header.version[1];
    handle->oskar_ver_patch = header.version[0];

    /* Read all tags in the stream. */
    for (i = 0; OSKAR_TRUE; ++i)
    {
        oskar_BinaryTag tag;
        size_t memcpy_size = 0;

        /* Check if we need to allocate more storage for the tag. */
        if (i % 10 == 0)
            oskar_binary_resize(handle, i + 10);

        /* Initialise the tag index data. */
        handle->extended[i] = 0;
        handle->data_type[i] = 0;
        handle->id_group[i] = 0;
        handle->id_tag[i] = 0;
        handle->name_group[i] = 0;
        handle->name_tag[i] = 0;
        handle->user_index[i] = 0;
        handle->data_offset_bytes[i] = 0;
        handle->data_size_bytes[i] = 0;
        handle->block_size_bytes[i] = 0;

        /* Try to read a tag, and end the loop if unsuccessful. */
        if (fread(&tag, sizeof(oskar_BinaryTag), 1, stream) != 1)
            break;

        /* If the bytes read are not a tag, then return an error. */
        if (tag.magic[0] != 'T' || tag.magic[1] != 'A' || tag.magic[2] != 'G' ||
                tag.magic[3] != 0)
        {
            *status = OSKAR_ERR_BINARY_FILE_INVALID;
            break;
        }

        /* Get the data type and IDs. */
        handle->data_type[i] = (int) tag.data_type;
        handle->id_group[i] = (int) tag.group.id;
        handle->id_tag[i] = (int) tag.tag.id;

        /* Copy out the index. */
        memcpy_size = MIN(sizeof(int), sizeof(tag.user_index));
        memcpy(&handle->user_index[i], tag.user_index, memcpy_size);

        /* Get the index in native byte order. */
        if (oskar_endian() != OSKAR_LITTLE_ENDIAN)
        {
            if (sizeof(int) == 2)
                oskar_endian_swap_2((char*)(&handle->user_index[i]));
            else if (sizeof(int) == 4)
                oskar_endian_swap_4((char*)(&handle->user_index[i]));
            else if (sizeof(int) == 8)
                oskar_endian_swap_8((char*)(&handle->user_index[i]));
        }

        /* Copy out the number of bytes in the block. */
        memcpy_size = MIN(sizeof(size_t), sizeof(tag.size_bytes));
        memcpy(&handle->block_size_bytes[i], tag.size_bytes, memcpy_size);

        /* Get the number of bytes in the block in native byte order. */
        if (oskar_endian() != OSKAR_LITTLE_ENDIAN)
        {
            if (sizeof(size_t) == 2)
                oskar_endian_swap_2((char*)(&handle->block_size_bytes[i]));
            else if (sizeof(size_t) == 4)
                oskar_endian_swap_4((char*)(&handle->block_size_bytes[i]));
            else if (sizeof(size_t) == 8)
                oskar_endian_swap_8((char*)(&handle->block_size_bytes[i]));
        }

        /* Store the data size. */
        handle->data_size_bytes[i] = handle->block_size_bytes[i];

        /* Check if the tag is extended. */
        if (tag.flags & (1 << 7))
        {
            size_t lgroup, ltag;

            /* Extended tag: set the extended flag. */
            handle->extended[i] = 1;

            /* Get the lengths of the strings. */
            lgroup = tag.group.bytes;
            ltag = tag.tag.bytes;

            /* Modify data size. */
            handle->data_size_bytes[i] -= (lgroup + ltag);

            /* Allocate memory for the tag names. */
            handle->name_group[i] = (char*) malloc(lgroup);
            handle->name_tag[i]   = (char*) malloc(ltag);

            /* Copy the tag names into the index. */
            if (fread(handle->name_group[i], 1, lgroup, stream) != lgroup)
                *status = OSKAR_ERR_BINARY_FILE_INVALID;
            if (fread(handle->name_tag[i], 1, ltag, stream) != ltag)
                *status = OSKAR_ERR_BINARY_FILE_INVALID;
            if (*status) break;
        }

        /* Store the current stream pointer as the data offset. */
        handle->data_offset_bytes[i] = ftell(stream);

        /* Increment stream pointer by data size. */
        fseek(stream, handle->data_size_bytes[i], SEEK_CUR);

        /* Save the number of tags read from the stream. */
        handle->num_tags = i + 1;
    }

    return handle;
}

static void oskar_binary_resize(oskar_Binary* handle, int m)
{
    handle->extended = (int*) realloc(handle->extended, m * sizeof(int));
    handle->data_type = (int*) realloc(handle->data_type, m * sizeof(int));
    handle->id_group = (int*) realloc(handle->id_group, m * sizeof(int));
    handle->id_tag = (int*) realloc(handle->id_tag, m * sizeof(int));
    handle->name_group = (char**) realloc(handle->name_group,
            m * sizeof(char*));
    handle->name_tag = (char**) realloc(handle->name_tag, m * sizeof(char*));
    handle->user_index = (int*) realloc(handle->user_index, m * sizeof(int));
    handle->data_offset_bytes = (long*) realloc(handle->data_offset_bytes,
            m * sizeof(long));
    handle->data_size_bytes = (size_t*) realloc(handle->data_size_bytes,
            m * sizeof(size_t));
    handle->block_size_bytes = (size_t*) realloc(handle->block_size_bytes,
            m * sizeof(size_t));
}

static void oskar_binary_write_header(FILE* stream, int* status)
{
    int version = OSKAR_VERSION;
    oskar_BinaryHeader header;
    char magic[] = "OSKARBIN";

    /* Construct binary header. */
    strcpy(header.magic, magic);
    header.bin_version = OSKAR_BINARY_FORMAT_VERSION;
    header.endian      = (char)oskar_endian();
    header.size_ptr    = (char)sizeof(void*);
    header.size_int    = (char)sizeof(int);
    header.size_long   = (char)sizeof(long);
    header.size_float  = (char)sizeof(float);
    header.size_double = (char)sizeof(double);

    /* Write OSKAR version data in little-endian format. */
    if (oskar_endian() != OSKAR_LITTLE_ENDIAN)
    {
        if (sizeof(int) == 4)
            oskar_endian_swap_4((char*)&version);
        else if (sizeof(int) == 8)
            oskar_endian_swap_8((char*)&version);
    }
    memcpy(header.version, &version, sizeof(header.version));

    /* Pad rest of header with zeros. */
    memset(header.reserved, 0, sizeof(header.reserved));

    /* Set stream pointer to beginning. */
    rewind(stream);

    /* Write header to stream. */
    if (fwrite(&header, sizeof(oskar_BinaryHeader), 1, stream) != 1)
        *status = OSKAR_ERR_FILE_IO;
}


static void oskar_binary_read_header(FILE* stream, oskar_BinaryHeader* header,
        int* status)
{
    /* Read the header from the stream. */
    rewind(stream);
    if (fread(header, sizeof(oskar_BinaryHeader), 1, stream) != 1)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Check if this is a valid header. */
    if (strncmp("OSKARBIN", header->magic, 8) != 0)
    {
        *status = OSKAR_ERR_BINARY_FILE_INVALID;
        return;
    }

    /* Check if the format is compatible. */
    if (OSKAR_BINARY_FORMAT_VERSION != (int)(header->bin_version))
    {
        *status = OSKAR_ERR_BINARY_VERSION_UNKNOWN;
        return;
    }

    /* Check if the architecture is compatible. */
    if (oskar_endian() != (int)(header->endian))
    {
        *status = OSKAR_ERR_BINARY_ENDIAN_MISMATCH;
        return;
    }

    /* Check size of data types. */
    if (sizeof(int) != (size_t)(header->size_int))
        *status = OSKAR_ERR_BINARY_INT_MISMATCH;
    if (sizeof(float) != (size_t)(header->size_float))
        *status = OSKAR_ERR_BINARY_FLOAT_MISMATCH;
    if (sizeof(double) != (size_t)(header->size_double))
        *status = OSKAR_ERR_BINARY_DOUBLE_MISMATCH;
}

#ifdef __cplusplus
}
#endif
