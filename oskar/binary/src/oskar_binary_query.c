/*
 * Copyright (c) 2012-2017, The University of Oxford
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
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_binary_num_tags(const oskar_Binary* handle)
{
    return handle->num_chunks;
}

int oskar_binary_tag_data_type(const oskar_Binary* handle, int tag_index)
{
    return tag_index < handle->num_chunks ?
            handle->data_type[tag_index] : 0;
}

size_t oskar_binary_tag_payload_size(const oskar_Binary* handle,
        int tag_index)
{
    return tag_index < handle->num_chunks ?
            handle->payload_size_bytes[tag_index] : 0;
}

int oskar_binary_query(const oskar_Binary* handle,
        unsigned char data_type, unsigned char id_group, unsigned char id_tag,
        int user_index, size_t* payload_size, int* status)
{
    int i;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Find the tag in the index. */
    for (i = handle->query_search_start; i < handle->num_chunks; ++i)
    {
        if (!(handle->extended[i]) &&
                ((handle->data_type[i] == (int) data_type) || (!data_type)) &&
                handle->id_group[i] == (int) id_group &&
                handle->id_tag[i] == (int) id_tag &&
                handle->user_index[i] == user_index)
        {
            /* Match found, so break. */
            break;
        }
    }

    /* Check if tag is not present. */
    if (i >= handle->num_chunks)
    {
        *status = OSKAR_ERR_BINARY_TAG_NOT_FOUND;
        return -1;
    }

    if (payload_size) *payload_size = handle->payload_size_bytes[i];
    return i;
}

int oskar_binary_query_ext(const oskar_Binary* handle,
        unsigned char data_type, const char* name_group, const char* name_tag,
        int user_index, size_t* payload_size, int* status)
{
    int i, lgroup, ltag;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Check that string lengths are within range. */
    lgroup = 1 + (int) strlen(name_group);
    ltag = 1 + (int) strlen(name_tag);
    if (lgroup > 255 || ltag > 255)
    {
        *status = OSKAR_ERR_BINARY_TAG_TOO_LONG;
        return -1;
    }

    /* Find the tag in the index. */
    for (i = handle->query_search_start; i < handle->num_chunks; ++i)
    {
        if (handle->extended[i] &&
                ((handle->data_type[i] == (int) data_type) || (!data_type)) &&
                handle->id_group[i] == (int) lgroup &&
                handle->id_tag[i] == (int) ltag &&
                handle->user_index[i] == user_index)
        {
            /* Possible match: check names. */
            if (strcmp(name_group, handle->name_group[i]))
                continue;
            if (strcmp(name_tag, handle->name_tag[i]))
                continue;

            /* Match found, so break. */
            break;
        }
    }

    /* Check if tag is not present. */
    if (i >= handle->num_chunks)
    {
        *status = OSKAR_ERR_BINARY_TAG_NOT_FOUND;
        return -1;
    }

    if (payload_size) *payload_size = handle->payload_size_bytes[i];
    return i;
}

void oskar_binary_set_query_search_start(oskar_Binary* handle, int start,
        int* status)
{
    if (start >= handle->num_chunks)
    {
        *status = OSKAR_ERR_BINARY_TAG_OUT_OF_RANGE;
        return;
    }
    handle->query_search_start = start;
}

#ifdef __cplusplus
}
#endif
