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
#include "utility/oskar_binary_file_read.h"
#include "utility/oskar_endian.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_binary_file_read(FILE* file, const oskar_BinaryTagIndex* index,
        unsigned char id, unsigned char id_user_1, unsigned char id_user_2,
        size_t block_size, void* data)
{
    int i;
    size_t size = 0, t;

    /* Find the tag ID in the index. */
    for (i = 0; i < index->num_tags; ++i)
    {
        if (id == index->tag[i].id &&
                id_user_1 == index->tag[i].id_user_1 &&
                id_user_2 == index->tag[i].id_user_2)
        {
            /* Match found, so break. */
            break;
        }
    }

    /* Check if tag is not present. */
    if (i == index->num_tags)
        return OSKAR_ERR_BINARY_TAG_NOT_FOUND;

    /* Get the number of bytes in the block in native byte order. */
    t = sizeof(size_t) < 8 ? sizeof(size_t) : 8;
    memcpy(&size, index->tag[i].size_bytes, t);
    if (oskar_endian() != OSKAR_LITTLE_ENDIAN)
    {
        if (sizeof(size_t) == 4)
            oskar_endian_swap_4((char*)&size);
        else if (sizeof(size_t) == 8)
            oskar_endian_swap_8((char*)&size);
    }

    /* Check that there is enough memory in the block. */
    if (block_size < size)
        return OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    /* Copy the data out of the file. */
    if (fseek(file, index->block_offset_bytes[i], SEEK_SET) != 0)
        return OSKAR_ERR_FILE_IO;
    if (fread(data, 1, size, file) != size)
        return OSKAR_ERR_FILE_IO;

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
