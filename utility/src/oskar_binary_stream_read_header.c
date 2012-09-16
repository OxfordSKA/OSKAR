/*
 * Copyright (c) 2012, The University of Oxford
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

void oskar_binary_stream_read_header(FILE* stream, oskar_BinaryHeader* header,
        int* status)
{
    /* Check all inputs. */
    if (!stream || !header || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Set stream pointer to beginning. */
    rewind(stream);

    /* Read the header from the stream. */
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
