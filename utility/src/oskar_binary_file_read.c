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

#include "utility/oskar_binary_file_read.h"
#include "utility/oskar_binary_stream_read_header.h"
#include "utility/oskar_binary_stream_read.h"
#include "utility/oskar_binary_tag_index_create.h"
#include "utility/oskar_endian.h"
#include "utility/oskar_Mem.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_binary_file_read(const char* filename,
        oskar_BinaryTagIndex** index, unsigned char id,
        unsigned char id_user_1, unsigned char id_user_2,
        unsigned char data_type, size_t data_size, void* data)
{
    FILE* stream;
    int err;

    /* Open the file for read. */
    stream = fopen(filename, "rb");

    /* Index the file if necessary. */
    if (*index == NULL)
    {
        err = oskar_binary_tag_index_create(index, stream);
        if (err)
        {
            fclose(stream);
            return err;
        }
    }

    /* Read the data. */
    err = oskar_binary_stream_read(stream, *index, id, id_user_1, id_user_2,
            data_type, data_size, data);

    /* Close the file. */
    fclose(stream);

    return err;
}

int oskar_binary_file_read_double(const char* filename,
        oskar_BinaryTagIndex** index, unsigned char id,
        unsigned char id_user_1, unsigned char id_user_2, double* value)
{
    return oskar_binary_file_read(filename, index, id, id_user_1, id_user_2,
            OSKAR_DOUBLE, sizeof(double), value);
}

int oskar_binary_file_read_int(const char* filename,
        oskar_BinaryTagIndex** index, unsigned char id,
        unsigned char id_user_1, unsigned char id_user_2, int* value)
{
    return oskar_binary_file_read(filename, index, id, id_user_1, id_user_2,
            OSKAR_INT, sizeof(int), value);
}

#ifdef __cplusplus
}
#endif
