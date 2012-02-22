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

#include "utility/oskar_BinaryHeader.h"
#include "utility/oskar_BinaryTag.h"
#include "utility/oskar_binary_file_read_header.h"
#include "utility/oskar_binary_tag_index_create.h"
#include "utility/oskar_endian.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_binary_tag_index_create(oskar_BinaryTagIndex* index,
        const char* filename)
{
    FILE* file;
    oskar_BinaryHeader header;
    int error, n;

    /* Open file for reading. */
    file = fopen(filename, "rb");

    /* Read header. */
    error = oskar_binary_file_read_header(file, &header);
    if (error) return error;

    /* Initialise tag index. */
    index->tag = 0;
    index->block_offset_bytes = 0;
    index->num_tags = 0;

    /* Read all tags in the file. */
    n = 0;
    for (;;)
    {
        size_t t, block_size;

        /* Check if we need to allocate more storage for the tag. */
        if (n % 10 == 0)
        {
            int m;

            /* New size of arrays. */
            m = n + 10;
            index->tag = realloc(index->tag, m * sizeof(oskar_BinaryTag));
            index->block_offset_bytes = realloc(index->block_offset_bytes,
                    m * sizeof(long));
        }

        /* Initialise the data block offset to 0. */
        index->block_offset_bytes[n] = 0;

        /* Try to read a tag, and end the loop if unsuccessful. */
        if (fread(&(index->tag[n]), sizeof(oskar_BinaryTag), 1, file) != 1)
            break;

        /* If the bytes read are not a tag, then return an error. */
        if (strcmp(index->tag[n].magic, "TAG") != 0)
        {
            fclose(file);
            return OSKAR_ERR_BAD_BINARY_FORMAT;
        }

        /* Store the current file pointer as the data block offset. */
        index->block_offset_bytes[n] = ftell(file);

        /* Get the number of bytes in the block in native byte order. */
        t = sizeof(size_t) < 8 ? sizeof(size_t) : 8;
        memcpy(&block_size, index->tag[n].size_bytes, t);
        if (oskar_endian() != OSKAR_LITTLE_ENDIAN)
        {
            if (sizeof(size_t) == 4)
                oskar_endian_swap_4((char*)&block_size);
            else if (sizeof(size_t) == 8)
                oskar_endian_swap_8((char*)&block_size);
        }

        /* Increment file pointer by block size. */
        fseek(file, block_size, SEEK_CUR);

        /* Increment tag counter. */
        n++;
    }

    /* Save the number of tags read from the file. */
    index->num_tags = n;

    /* Close file. */
    fclose(file);

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
