/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "binary/oskar_binary.h"
#include "binary/private_binary.h"
#include "binary/oskar_endian.h"
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_binary_write(oskar_Binary* handle, unsigned char data_type,
        unsigned char id_group, unsigned char id_tag, int user_index,
        size_t data_size, const void* data, int* status)
{
    oskar_BinaryTag tag;
    size_t block_size = 0;
    unsigned long crc = 0;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check file was opened for writing. */
    if (handle->open_mode != 'w' && handle->open_mode != 'a')
    {
        *status = OSKAR_ERR_BINARY_NOT_OPEN_FOR_WRITE;
        return;
    }

    /* Initialise the tag. */
    tag.magic[0] = 'T';
    tag.magic[1] = 0x40 + OSKAR_BINARY_FORMAT_VERSION;
    tag.magic[2] = 'G';
    tag.magic[3] = 0;
    memset(tag.size_bytes, 0, sizeof(tag.size_bytes));
    memset(tag.user_index, 0, sizeof(tag.user_index));

    /* Set the size of the payload element. */
    if (data_type & OSKAR_CHAR)
    {
        tag.magic[3] = sizeof(char);
    }
    else if (data_type & OSKAR_INT)
    {
        tag.magic[3] = sizeof(int);
    }
    else if (data_type & OSKAR_SINGLE)
    {
        tag.magic[3] = sizeof(float);
    }
    else if (data_type & OSKAR_DOUBLE)
    {
        tag.magic[3] = sizeof(double);
    }
    else
    {
        *status = OSKAR_ERR_BINARY_TYPE_UNKNOWN;
        return;
    }
    if (data_type & OSKAR_COMPLEX)
    {
        tag.magic[3] *= 2;
    }
    if (data_type & OSKAR_MATRIX)
    {
        tag.magic[3] *= 4;
    }

    /* Set up the tag identifiers */
    tag.flags = 0;
    tag.flags |= (1 << 6); /* Set bit 6 to indicate CRC-32C code added. */
    tag.data_type = data_type;
    tag.group.id = id_group;
    tag.tag.id = id_tag;

    /* Get the number of bytes in the block and user index in
     * little-endian byte order (add 4 for CRC). */
    block_size = data_size + 4;
    if (sizeof(size_t) != 4 && sizeof(size_t) != 8)
    {
        *status = OSKAR_ERR_BINARY_FORMAT_BAD;
        return;
    }
    if (oskar_endian() != OSKAR_LITTLE_ENDIAN)
    {
        oskar_endian_swap(&block_size, sizeof(size_t));
        oskar_endian_swap(&user_index, sizeof(int));
    }

    /* Copy user index and block size to the tag, as little endian values. */
    memcpy(tag.user_index, &user_index, sizeof(int));
    memcpy(tag.size_bytes, &block_size, sizeof(size_t));

    /* Tag is complete at this point, so calculate CRC. */
    crc = oskar_crc_compute(handle->crc_data, &tag, sizeof(oskar_BinaryTag));
    crc = oskar_crc_update(handle->crc_data, crc, data, data_size);
    if (oskar_endian() != OSKAR_LITTLE_ENDIAN)
    {
        oskar_endian_swap(&crc, sizeof(unsigned long));
    }

    /* Write the tag to the file. */
    if (fwrite(&tag, sizeof(oskar_BinaryTag), 1, handle->stream) != 1)
    {
        *status = OSKAR_ERR_BINARY_WRITE_FAIL;
        return;
    }

    /* Check there is data to write. */
    if (data && data_size > 0)
    {
        /* Write the data to the file. */
        if (fwrite(data, 1, data_size, handle->stream) != data_size)
        {
            *status = OSKAR_ERR_BINARY_WRITE_FAIL;
            return;
        }
    }

    /* Write the 4-byte CRC-32C code. */
    if (fwrite(&crc, 4, 1, handle->stream) != 1)
    {
        *status = OSKAR_ERR_BINARY_WRITE_FAIL;
    }
}

void oskar_binary_write_double(oskar_Binary* handle, unsigned char id_group,
        unsigned char id_tag, int user_index, double value, int* status)
{
    oskar_binary_write(handle, OSKAR_DOUBLE, id_group, id_tag,
            user_index, sizeof(double), &value, status);
}

void oskar_binary_write_int(oskar_Binary* handle, unsigned char id_group,
        unsigned char id_tag, int user_index, int value, int* status)
{
    oskar_binary_write(handle, OSKAR_INT, id_group, id_tag,
            user_index, sizeof(int), &value, status);
}

void oskar_binary_write_ext(oskar_Binary* handle, unsigned char data_type,
        const char* name_group, const char* name_tag, int user_index,
        size_t data_size, const void* data, int* status)
{
    oskar_BinaryTag tag;
    size_t block_size = 0, lgroup = 0, ltag = 0;
    unsigned long crc = 0;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check file was opened for writing. */
    if (handle->open_mode != 'w' && handle->open_mode != 'a')
    {
        *status = OSKAR_ERR_BINARY_NOT_OPEN_FOR_WRITE;
        return;
    }

    /* Initialise the tag. */
    tag.magic[0] = 'T';
    tag.magic[1] = 0x40 + OSKAR_BINARY_FORMAT_VERSION;
    tag.magic[2] = 'G';
    tag.magic[3] = 0;
    memset(tag.size_bytes, 0, sizeof(tag.size_bytes));
    memset(tag.user_index, 0, sizeof(tag.user_index));

    /* Set the size of the payload element. */
    if (data_type & OSKAR_CHAR)
    {
        tag.magic[3] = sizeof(char);
    }
    else if (data_type & OSKAR_INT)
    {
        tag.magic[3] = sizeof(int);
    }
    else if (data_type & OSKAR_SINGLE)
    {
        tag.magic[3] = sizeof(float);
    }
    else if (data_type & OSKAR_DOUBLE)
    {
        tag.magic[3] = sizeof(double);
    }
    else
    {
        *status = OSKAR_ERR_BINARY_TYPE_UNKNOWN;
        return;
    }
    if (data_type & OSKAR_COMPLEX)
    {
        tag.magic[3] *= 2;
    }
    if (data_type & OSKAR_MATRIX)
    {
        tag.magic[3] *= 4;
    }

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
    tag.flags |= (1 << 6); /* Set bit 6 to indicate CRC-32C code added. */
    tag.data_type = data_type;
    tag.group.bytes = 1 + (unsigned char)lgroup;
    tag.tag.bytes = 1 + (unsigned char)ltag;

    /* Get the number of bytes in the block and user index in
     * little-endian byte order (add 4 for CRC). */
    block_size = data_size + tag.group.bytes + tag.tag.bytes + 4;
    if (sizeof(size_t) != 4 && sizeof(size_t) != 8)
    {
        *status = OSKAR_ERR_BINARY_FORMAT_BAD;
        return;
    }
    if (oskar_endian() != OSKAR_LITTLE_ENDIAN)
    {
        oskar_endian_swap(&block_size, sizeof(size_t));
        oskar_endian_swap(&user_index, sizeof(int));
    }

    /* Copy user index and block size to the tag, as little endian values. */
    memcpy(tag.user_index, &user_index, sizeof(int));
    memcpy(tag.size_bytes, &block_size, sizeof(size_t));

    /* Tag is complete at this point, so calculate CRC. */
    crc = oskar_crc_compute(handle->crc_data, &tag, sizeof(oskar_BinaryTag));
    crc = oskar_crc_update(handle->crc_data, crc, name_group, tag.group.bytes);
    crc = oskar_crc_update(handle->crc_data, crc, name_tag, tag.tag.bytes);
    crc = oskar_crc_update(handle->crc_data, crc, data, data_size);
    if (oskar_endian() != OSKAR_LITTLE_ENDIAN)
    {
        oskar_endian_swap(&crc, sizeof(unsigned long));
    }

    /* Write the tag to the file. */
    if (fwrite(&tag, sizeof(oskar_BinaryTag), 1, handle->stream) != 1)
    {
        *status = OSKAR_ERR_BINARY_WRITE_FAIL;
        return;
    }

    /* Write the group name and tag name to the file. */
    if (fwrite(name_group, tag.group.bytes, 1, handle->stream) != 1)
    {
        *status = OSKAR_ERR_BINARY_WRITE_FAIL;
        return;
    }
    if (fwrite(name_tag, tag.tag.bytes, 1, handle->stream) != 1)
    {
        *status = OSKAR_ERR_BINARY_WRITE_FAIL;
        return;
    }

    /* Check there is data to write. */
    if (data && data_size > 0)
    {
        /* Write the data to the file. */
        if (fwrite(data, 1, data_size, handle->stream) != data_size)
        {
            *status = OSKAR_ERR_BINARY_WRITE_FAIL;
            return;
        }
    }

    /* Write the 4-byte CRC-32C code. */
    if (fwrite(&crc, 4, 1, handle->stream) != 1)
    {
        *status = OSKAR_ERR_BINARY_WRITE_FAIL;
    }
}

void oskar_binary_write_ext_double(oskar_Binary* handle, const char* name_group,
        const char* name_tag, int user_index, double value, int* status)
{
    oskar_binary_write_ext(handle, OSKAR_DOUBLE, name_group,
            name_tag, user_index, sizeof(double), &value, status);
}

void oskar_binary_write_ext_int(oskar_Binary* handle, const char* name_group,
        const char* name_tag, int user_index, int value, int* status)
{
    oskar_binary_write_ext(handle, OSKAR_INT, name_group,
            name_tag, user_index, sizeof(int), &value, status);
}

#ifdef __cplusplus
}
#endif
