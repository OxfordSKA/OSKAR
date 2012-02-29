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

#include "utility/oskar_binary_stream_read.h"
#include "utility/oskar_binary_tag_index_create.h"
#include "utility/oskar_binary_tag_index_query.h"
#include "utility/oskar_mem_binary_file_read.h"
#include "utility/oskar_mem_binary_stream_read.h"
#include "utility/oskar_mem_copy.h"
#include "utility/oskar_mem_element_size.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_realloc.h"

#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_mem_binary_file_read(oskar_Mem* mem, const char* filename,
        oskar_BinaryTagIndex** index, unsigned char id_group,
        unsigned char id_tag, int user_index)
{
    int err;
    FILE* stream;

    /* Sanity check on inputs. */
    if (mem == NULL || filename == NULL || index == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Open the input file. */
    stream = fopen(filename, "rb");
    if (stream == NULL)
        return OSKAR_ERR_FILE_IO;

    /* Read from the file. */
    err = oskar_mem_binary_stream_read(mem, stream, index,
            id_group, id_tag, user_index);

    /* Close the input file and check for errors. */
    fclose(stream);

    return err;
}

int oskar_mem_binary_file_read_ext(oskar_Mem* mem, const char* filename,
        oskar_BinaryTagIndex** index, const char* name_group,
        const char* name_tag, int user_index)
{
    int err;
    FILE* stream;

    /* Sanity check on inputs. */
    if (mem == NULL || filename == NULL || index == NULL ||
            name_group == NULL || name_tag == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Open the input file. */
    stream = fopen(filename, "rb");
    if (stream == NULL)
        return OSKAR_ERR_FILE_IO;

    /* Read from the file. */
    err = oskar_mem_binary_stream_read_ext(mem, stream, index,
            name_group, name_tag, user_index);

    /* Close the input file and check for errors. */
    fclose(stream);

    return err;
}

#ifdef __cplusplus
}
#endif
