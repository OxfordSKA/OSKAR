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

#include "utility/oskar_BinaryTag.h"
#include "utility/oskar_binary_file_write.h"
#include "utility/oskar_binary_stream_read_header.h"
#include "utility/oskar_binary_stream_write_header.h"
#include "utility/oskar_binary_stream_write.h"
#include "utility/oskar_binary_stream_write_std_metadata.h"
#include "utility/oskar_endian.h"
#include "utility/oskar_Mem.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_binary_file_write(const char* filename, unsigned char data_type,
        const char* name_group, const char* name_tag, int user_index,
        size_t data_size, const void* data)
{
    FILE* stream;
    int err;

    /* Sanity check on inputs. */
    if (filename == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Open the file for read and append. */
    stream = fopen(filename, "a+b");

    /* Check if the file is empty. */
    fseek(stream, 0, SEEK_END);
    if (ftell(stream) == 0)
    {
        /* If the file is empty, then write the header. */
        err = oskar_binary_stream_write_header(stream);
        if (err)
        {
            fclose(stream);
            return err;
        }

        /* Write standard metadata. */
        err = oskar_binary_stream_write_std_metadata(stream);
        if (err)
        {
            fclose(stream);
            return err;
        }
    }
    else
    {
        /* If the file is not empty, then check the header. */
        oskar_BinaryHeader header;
        err = oskar_binary_stream_read_header(stream, &header);
        if (err)
        {
            fclose(stream);
            return err;
        }

        /* Seek to end of file. */
        fseek(stream, 0, SEEK_END);
    }

    /* Write the data. */
    err = oskar_binary_stream_write(stream, data_type,
            name_group, name_tag, user_index, data_size, data);

    /* Close the file. */
    fclose(stream);

    return err;
}

int oskar_binary_file_write_double(const char* filename,
        const char* name_group, const char* name_tag, int user_index,
        double value)
{
    return oskar_binary_file_write(filename, OSKAR_DOUBLE, name_group,
            name_tag, user_index, sizeof(double), &value);
}

int oskar_binary_file_write_int(const char* filename,
        const char* name_group, const char* name_tag, int user_index,
        int value)
{
    return oskar_binary_file_write(filename, OSKAR_INT, name_group,
            name_tag, user_index, sizeof(int), &value);
}

int oskar_binary_file_write_std(const char* filename, unsigned char data_type,
        unsigned char id_group, unsigned char id_tag, int user_index,
        size_t data_size, const void* data)
{
    FILE* stream;
    int err;

    /* Sanity check on inputs. */
    if (filename == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Open the file for read and append. */
    stream = fopen(filename, "a+b");

    /* Check if the file is empty. */
    fseek(stream, 0, SEEK_END);
    if (ftell(stream) == 0)
    {
        /* If the file is empty, then write the header. */
        err = oskar_binary_stream_write_header(stream);
        if (err)
        {
            fclose(stream);
            return err;
        }

        /* Write standard metadata. */
        err = oskar_binary_stream_write_std_metadata(stream);
        if (err)
        {
            fclose(stream);
            return err;
        }
    }
    else
    {
        /* If the file is not empty, then check the header. */
        oskar_BinaryHeader header;
        err = oskar_binary_stream_read_header(stream, &header);
        if (err)
        {
            fclose(stream);
            return err;
        }

        /* Seek to end of file. */
        fseek(stream, 0, SEEK_END);
    }

    /* Write the data. */
    err = oskar_binary_stream_write_std(stream, data_type,
            id_group, id_tag, user_index, data_size, data);

    /* Close the file. */
    fclose(stream);

    return err;
}

int oskar_binary_file_write_std_double(const char* filename,
        unsigned char id_group, unsigned char id_tag, int user_index,
        double value)
{
    return oskar_binary_file_write_std(filename, OSKAR_DOUBLE, id_group,
            id_tag, user_index, sizeof(double), &value);
}

int oskar_binary_file_write_std_int(const char* filename,
        unsigned char id_group, unsigned char id_tag, int user_index,
        int value)
{
    return oskar_binary_file_write_std(filename, OSKAR_INT, id_group,
            id_tag, user_index, sizeof(int), &value);
}

#ifdef __cplusplus
}
#endif
