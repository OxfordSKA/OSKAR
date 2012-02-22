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
#include "utility/oskar_endian.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_binary_file_write(FILE* file, unsigned char id,
        unsigned char id_user_1, unsigned char id_user_2,
        unsigned char data_type, size_t bytes, const void* data)
{
    oskar_BinaryTag tag;
    size_t size_bytes;

    /* Initialise the tag. */
    char magic[] = "TAG";
    size_bytes = bytes;
    strcpy(tag.magic, magic);
    memset(tag.size_bytes, 0, sizeof(tag.size_bytes));

    /* Set up the tag identifiers */
    tag.id = id;
    tag.data_type = data_type;
    tag.id_user_1 = id_user_1;
    tag.id_user_2 = id_user_2;

    /* Write the block size in bytes as little endian. */
    if (sizeof(size_t) != 4 && sizeof(size_t) != 8)
    {
        return OSKAR_ERR_BAD_BINARY_FORMAT;
    }
    if (oskar_endian() != OSKAR_LITTLE_ENDIAN)
    {
        if (sizeof(size_t) == 4)
            oskar_endian_swap_4((char*)&size_bytes);
        else if (sizeof(size_t) == 8)
            oskar_endian_swap_8((char*)&size_bytes);
    }
    memcpy(tag.size_bytes, &size_bytes, sizeof(size_t));

    /* Write the tag to the file. */
    if (fwrite(&tag, sizeof(oskar_BinaryTag), 1, file) != 1)
        return OSKAR_ERR_FILE_IO;

    /* Write the data to the file. */
    if (fwrite(data, 1, bytes, file) != bytes)
        return OSKAR_ERR_FILE_IO;

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
