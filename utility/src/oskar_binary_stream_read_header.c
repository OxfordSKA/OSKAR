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

#include "utility/oskar_binary_header_version.h"
#include "utility/oskar_binary_stream_read_header.h"
#include "utility/oskar_endian.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_binary_stream_read_header(FILE* file, oskar_BinaryHeader* header)
{
    /* Set file pointer to beginning. */
    rewind(file);

    /* Read the header from the file. */
    if (fread(header, sizeof(oskar_BinaryHeader), 1, file) != 1)
        return OSKAR_ERR_FILE_IO;

    /* Check if this is a valid header. */
    if (strcmp("OSKARBIN", header->magic) != 0)
    {
        fprintf(stderr, "Error: Invalid OSKAR binary file.\n");
        return OSKAR_ERR_BAD_BINARY_FORMAT;
    }

    /* Check if the format is compatible. */
    if (OSKAR_BINARY_FORMAT_VERSION != (int)(header->bin_version))
    {
        fprintf(stderr, "Error: Unknown OSKAR binary file format.\n");
        return OSKAR_ERR_BAD_BINARY_FORMAT;
    }

    /* Check if the architecture is compatible. */
    if (oskar_endian() != (int)(header->endian))
    {
        fprintf(stderr, "Error: Incompatible architecture "
                "(wrong data byte ordering).\n");
        return OSKAR_ERR_BAD_BINARY_FORMAT;
    }

    /* Check size of pointer type. */
    if (sizeof(void*) < (size_t)(header->size_ptr))
    {
        fprintf(stderr, "Warning: (sizeof(void*) < file sizeof(void*)).\n");
    }

    /* Check size of data types. */
    if (sizeof(int) != (size_t)(header->size_int))
    {
        fprintf(stderr, "Error: Incompatible architecture "
                "(sizeof(int) != file sizeof(int)).\n");
        return OSKAR_ERR_BAD_BINARY_FORMAT;
    }
    if (sizeof(float) != (size_t)(header->size_float))
    {
        fprintf(stderr, "Error: Incompatible architecture "
                "(sizeof(float) != file sizeof(float)).\n");
        return OSKAR_ERR_BAD_BINARY_FORMAT;
    }
    if (sizeof(double) != (size_t)(header->size_double))
    {
        fprintf(stderr, "Error: Incompatible architecture "
                "(sizeof(double) != file sizeof(double)).\n");
        return OSKAR_ERR_BAD_BINARY_FORMAT;
    }

    /* Check OSKAR version. */
    if ((int)OSKAR_VERSION != oskar_binary_header_version(header))
    {
        fprintf(stderr, "Warning: OSKAR_VERSION mismatch.\n");
    }

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
