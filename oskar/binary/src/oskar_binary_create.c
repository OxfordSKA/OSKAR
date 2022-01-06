/*
 * Copyright (c) 2012-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "binary/oskar_binary.h"
#include "binary/oskar_endian.h"
#include "binary/private_binary.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef _MSC_VER
#include <sys/types.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

static void oskar_binary_resize(oskar_Binary* handle, int m);
static void oskar_binary_read_header(FILE* stream, oskar_BinaryHeader* header,
        int* status);
static void oskar_binary_write_header(FILE* stream, oskar_BinaryHeader* header,
        int* status);

#ifdef _MSC_VER
#define FTELL _ftelli64
#else
#define FTELL ftello
#endif

oskar_Binary* oskar_binary_create(const char* filename, char mode, int* status)
{
    oskar_Binary* handle = 0;
    oskar_BinaryHeader header;
    FILE* stream = 0;
    int i = 0;

    /* Initialise the header. This doesn't actually need to happen here,
     * since it will be done when the header is read or written,
     * but the linter should be happier with it. */
    memset(&header, 0, sizeof(oskar_BinaryHeader));
    header.bin_version = OSKAR_BINARY_FORMAT_VERSION;

    /* Open the file and check or write the header, depending on the mode. */
    if (mode == 'r')
    {
        stream = fopen(filename, "rb");
        if (!stream)
        {
            *status = OSKAR_ERR_BINARY_OPEN_FAIL;
            return 0;
        }
        oskar_binary_read_header(stream, &header, status);
        if (*status)
        {
            fclose(stream);
            return 0;
        }
    }
    else if (mode == 'w')
    {
        stream = fopen(filename, "wb");
        if (!stream)
        {
            *status = OSKAR_ERR_BINARY_OPEN_FAIL;
            return 0;
        }
        oskar_binary_write_header(stream, &header, status);
    }
    else if (mode == 'a')
    {
        stream = fopen(filename, "a+b");
        if (!stream)
        {
            *status = OSKAR_ERR_BINARY_OPEN_FAIL;
            return 0;
        }

        /* Write header only if the file is empty. */
        fseek(stream, 0, SEEK_END);
        if (FTELL(stream) == 0)
        {
            oskar_binary_write_header(stream, &header, status);
        }
    }
    else
    {
        *status = OSKAR_ERR_BINARY_OPEN_FAIL;
        return 0;
    }

    /* Allocate index and store the stream handle. */
    handle = (oskar_Binary*) calloc(1, sizeof(oskar_Binary));
    handle->stream = stream;
    handle->open_mode = mode;

    /* Create the CRC lookup tables. */
    handle->crc_data = oskar_crc_create(OSKAR_CRC_32C);

    /* Store the contents of the header for later use. */
    handle->bin_version = header.bin_version;

    /* Finish if writing. */
    if (mode == 'w')
    {
        return handle;
    }

    /* Read all tags in the stream. */
    for (i = 0;; ++i)
    {
        oskar_BinaryTag tag;
        unsigned long crc = 0;
        int format_version = 0, element_size = 0;
        size_t block_size = 0, memcpy_size = 0;

        /* Try to read a tag, and end the loop if unsuccessful. */
        if (fread(&tag, sizeof(oskar_BinaryTag), 1, stream) != 1)
        {
            break;
        }

        /* If the bytes read are not a tag, or the reserved flag bits
         * are not zero, then return an error. */
        if (tag.magic[0] != 'T' || tag.magic[2] != 'G'
                || (tag.flags & 0x1F) != 0)
        {
            *status = OSKAR_ERR_BINARY_FILE_INVALID;
            break;
        }

        /* Get the binary format version. */
        format_version = tag.magic[1] - 0x40;
        if (format_version < 1 || format_version > OSKAR_BINARY_FORMAT_VERSION)
        {
            *status = OSKAR_ERR_BINARY_VERSION_UNKNOWN;
            break;
        }

        /* Additional checks if format version > 1. */
        if (format_version > 1)
        {
            /* Check system byte order is compatible. */
            if (oskar_endian() && !(tag.flags & (1 << 5)))
            {
                *status = OSKAR_ERR_BINARY_ENDIAN_MISMATCH;
                break;
            }

            /* Check data size is compatible. */
            element_size = tag.magic[3];
            if (tag.data_type & OSKAR_MATRIX)
            {
                element_size /= 4;
            }
            if (tag.data_type & OSKAR_COMPLEX)
            {
                element_size /= 2;
            }
            if (tag.data_type & OSKAR_CHAR)
            {
                if (element_size != sizeof(char))
                {
                    *status = OSKAR_ERR_BINARY_FORMAT_BAD;
                }
            }
            else if (tag.data_type & OSKAR_INT)
            {
                if (element_size != sizeof(int))
                {
                    *status = OSKAR_ERR_BINARY_INT_UNKNOWN;
                }
            }
            else if (tag.data_type & OSKAR_SINGLE)
            {
                if (element_size != sizeof(float))
                {
                    *status = OSKAR_ERR_BINARY_FLOAT_UNKNOWN;
                }
            }
            else if (tag.data_type & OSKAR_DOUBLE)
            {
                if (element_size != sizeof(double))
                {
                    *status = OSKAR_ERR_BINARY_DOUBLE_UNKNOWN;
                }
            }
            else
            {
                *status = OSKAR_ERR_BINARY_TYPE_UNKNOWN;
            }
        }

        /* Check if we need to allocate more storage for the tag data. */
        if (i % 10 == 0)
        {
            oskar_binary_resize(handle, i + 10);
        }

        /* Initialise the tag index data. */
        handle->extended[i] = 0;
        handle->data_type[i] = 0;
        handle->id_group[i] = 0;
        handle->id_tag[i] = 0;
        handle->name_group[i] = 0;
        handle->name_tag[i] = 0;
        handle->user_index[i] = 0;
        handle->payload_offset_bytes[i] = 0;
        handle->payload_size_bytes[i] = 0;
        handle->crc[i] = 0;
        handle->crc_header[i] = 0;

        /* Start computing the CRC code. */
        crc = oskar_crc_compute(handle->crc_data, &tag,
                sizeof(oskar_BinaryTag));

        /* Store the data type and IDs. */
        handle->data_type[i] = (int) tag.data_type;
        handle->id_group[i] = (int) tag.group.id;
        handle->id_tag[i] = (int) tag.tag.id;

        /* Store the index in native byte order. */
        memcpy_size = MIN(sizeof(int), sizeof(tag.user_index));
        memcpy(&handle->user_index[i], tag.user_index, memcpy_size);
        if (oskar_endian() != OSKAR_LITTLE_ENDIAN)
        {
            oskar_endian_swap(&handle->user_index[i], sizeof(int));
        }

        /* Store the number of bytes in the block in native byte order. */
        memcpy_size = MIN(sizeof(size_t), sizeof(tag.size_bytes));
        memcpy(&block_size, tag.size_bytes, memcpy_size);
        if (oskar_endian() != OSKAR_LITTLE_ENDIAN)
        {
            oskar_endian_swap(&block_size, sizeof(size_t));
        }

        /* Set payload size to block size, minus 4 bytes if CRC-32 present. */
        handle->payload_size_bytes[i] = block_size;
        handle->payload_size_bytes[i] -= (tag.flags & (1 << 6) ? 4 : 0);

        /* Check if the tag is extended. */
        if (tag.flags & (1 << 7))
        {
            /* Extended tag: set the extended flag. */
            handle->extended[i] = 1;

            /* Reduce payload size by sum of length of tag names. */
            handle->payload_size_bytes[i] -= (tag.group.bytes + tag.tag.bytes);

            /* Allocate memory for the tag names. */
            handle->name_group[i] = (char*) malloc(tag.group.bytes);
            handle->name_tag[i]   = (char*) malloc(tag.tag.bytes);

            /* Store the tag names. */
            if (fread(handle->name_group[i], tag.group.bytes, 1, stream) != 1)
            {
                *status = OSKAR_ERR_BINARY_FILE_INVALID;
            }
            if (fread(handle->name_tag[i], tag.tag.bytes, 1, stream) != 1)
            {
                *status = OSKAR_ERR_BINARY_FILE_INVALID;
            }
            if (*status) break;

            /* Update the CRC code. */
            crc = oskar_crc_update(handle->crc_data, crc,
                    handle->name_group[i], tag.group.bytes);
            crc = oskar_crc_update(handle->crc_data, crc,
                    handle->name_tag[i], tag.tag.bytes);
        }

        /* Store the current stream pointer as the payload offset. */
        const int64_t cur_pos = (int64_t) FTELL(stream);
        if (cur_pos == -1)
        {
            *status = OSKAR_ERR_BINARY_READ_FAIL;
            break;
        }
        handle->payload_offset_bytes[i] = cur_pos;

        /* Increment stream pointer by payload size. */
#ifdef _MSC_VER
        if (_fseeki64(stream, handle->payload_size_bytes[i], SEEK_CUR))
#else
        if (fseeko(stream, (off_t) handle->payload_size_bytes[i], SEEK_CUR))
#endif
        {
            *status = OSKAR_ERR_BINARY_SEEK_FAIL;
            break;
        }

        /* Store header CRC code and get file CRC code in native byte order. */
        handle->crc_header[i] = crc;
        if (tag.flags & (1 << 6))
        {
            if (fread(&handle->crc[i], 4, 1, stream) != 1)
            {
                *status = OSKAR_ERR_BINARY_READ_FAIL;
                break;
            }

            if (oskar_endian() != OSKAR_LITTLE_ENDIAN)
            {
                oskar_endian_swap(&handle->crc[i], sizeof(unsigned long));
            }
        }

        /* Save the number of tags read from the stream. */
        handle->num_chunks = i + 1;
    }

    return handle;
}

static void oskar_binary_resize(oskar_Binary* handle, int m)
{
    handle->extended = (int*) realloc(handle->extended, m * sizeof(int));
    handle->data_type = (int*) realloc(handle->data_type, m * sizeof(int));
    handle->id_group = (int*) realloc(handle->id_group, m * sizeof(int));
    handle->id_tag = (int*) realloc(handle->id_tag, m * sizeof(int));
    handle->name_group = (char**) realloc(
            handle->name_group, m * sizeof(char*));
    handle->name_tag = (char**) realloc(handle->name_tag, m * sizeof(char*));
    handle->user_index = (int*) realloc(handle->user_index, m * sizeof(int));
    handle->payload_offset_bytes = (int64_t*) realloc(
            handle->payload_offset_bytes, m * sizeof(int64_t));
    handle->payload_size_bytes = (size_t*) realloc(
            handle->payload_size_bytes, m * sizeof(size_t));
    handle->crc = (unsigned long*) realloc(
            handle->crc, m * sizeof(unsigned long));
    handle->crc_header = (unsigned long*) realloc(
            handle->crc_header, m * sizeof(unsigned long));
}

static void oskar_binary_write_header(FILE* stream, oskar_BinaryHeader* header,
        int* status)
{
    const char magic[] = "OSKARBIN";

    /* Construct binary header. */
    memset(header, 0, sizeof(oskar_BinaryHeader));
    memcpy(header->magic, magic, sizeof(magic));
    header->bin_version = OSKAR_BINARY_FORMAT_VERSION;

    /* Write header to stream. */
    rewind(stream);
    if (fwrite(header, sizeof(oskar_BinaryHeader), 1, stream) != 1)
    {
        *status = OSKAR_ERR_BINARY_WRITE_FAIL;
    }
}


static void oskar_binary_read_header(FILE* stream, oskar_BinaryHeader* header,
        int* status)
{
    /* Read the header from the stream. */
    rewind(stream);
    if (fread(header, sizeof(oskar_BinaryHeader), 1, stream) != 1)
    {
        *status = OSKAR_ERR_BINARY_READ_FAIL;
        return;
    }

    /* Check if this is a valid header. */
    if (strncmp("OSKARBIN", header->magic, 8) != 0)
    {
        *status = OSKAR_ERR_BINARY_FILE_INVALID;
        return;
    }

    /* Check if the format is compatible. */
    if ((int)(header->bin_version) > OSKAR_BINARY_FORMAT_VERSION)
    {
        *status = OSKAR_ERR_BINARY_VERSION_UNKNOWN;
        return;
    }

    if (header->bin_version == 1)
    {
        /* Check if the architecture is compatible. */
        if (oskar_endian() != (int)(header->endian))
        {
            *status = OSKAR_ERR_BINARY_ENDIAN_MISMATCH;
            return;
        }

        /* Check size of data types. */
        if (sizeof(int) != (size_t)(header->size_int))
        {
            *status = OSKAR_ERR_BINARY_INT_UNKNOWN;
        }
        if (sizeof(float) != (size_t)(header->size_float))
        {
            *status = OSKAR_ERR_BINARY_FLOAT_UNKNOWN;
        }
        if (sizeof(double) != (size_t)(header->size_double))
        {
            *status = OSKAR_ERR_BINARY_DOUBLE_UNKNOWN;
        }
    }
}

#ifdef __cplusplus
}
#endif
