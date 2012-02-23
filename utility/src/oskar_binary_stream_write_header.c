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
#include "utility/oskar_binary_stream_write_header.h"
#include "utility/oskar_endian.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_binary_stream_write_header(FILE* stream)
{
    /* Construct binary header. */
    int version = OSKAR_VERSION;
    oskar_BinaryHeader header;
    char magic[] = "OSKARBIN";
    strcpy(header.magic, magic);
    header.bin_version = OSKAR_BINARY_FORMAT_VERSION;
    header.endian      = (char)oskar_endian();
    header.size_ptr    = (char)sizeof(void*);
    header.size_int    = (char)sizeof(int);
    header.size_long   = (char)sizeof(long);
    header.size_float  = (char)sizeof(float);
    header.size_double = (char)sizeof(double);

    /* Write OSKAR version data in little-endian format. */
    if (oskar_endian() != OSKAR_LITTLE_ENDIAN)
    {
        if (sizeof(int) == 4)
            oskar_endian_swap_4((char*)&version);
        else if (sizeof(int) == 8)
            oskar_endian_swap_8((char*)&version);
    }
    memcpy(header.version, &version, sizeof(header.version));

    /* Pad rest of header with zeros. */
    memset(header.reserved, 0, sizeof(header.reserved));

    /* Set stream pointer to beginning. */
    rewind(stream);

    /* Write header to stream. */
    if (fwrite(&header, sizeof(oskar_BinaryHeader), 1, stream) != 1)
        return OSKAR_ERR_FILE_IO;

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
