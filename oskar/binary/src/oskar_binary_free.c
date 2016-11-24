/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include "binary/oskar_binary.h"
#include "binary/private_binary.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_binary_free(oskar_Binary* handle)
{
    int i;

    /* Check if structure exists. */
    if (!handle) return;

    /* Close the file. */
    if (handle->stream)
        fclose(handle->stream);

    /* Free string data. */
    for (i = 0; i < handle->num_chunks; ++i)
    {
        free(handle->name_group[i]);
        free(handle->name_tag[i]);
    }

    /* Free arrays. */
    free(handle->extended);
    free(handle->data_type);
    free(handle->id_group);
    free(handle->id_tag);
    free(handle->name_group);
    free(handle->name_tag);
    free(handle->user_index);
    free(handle->payload_offset_bytes);
    free(handle->payload_size_bytes);
    free(handle->block_size_bytes);
    free(handle->crc);
    free(handle->crc_header);

    /* Free the CRC data. */
    oskar_crc_free(handle->crc_data);

    /* Free the structure itself. */
    free(handle);
}

#ifdef __cplusplus
}
#endif
