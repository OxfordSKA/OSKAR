/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#include <private_binary.h>
#include <oskar_binary_query.h>
#include <oskar_endian.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_binary_query(const oskar_Binary* index,
        unsigned char data_type, unsigned char id_group, unsigned char id_tag,
        int user_index, size_t* data_size, long* data_offset, int* status)
{
    int i;

    /* Check all inputs. */
    if (!index || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Find the tag in the index. */
    for (i = 0; i < index->num_tags; ++i)
    {
        if (!(index->extended[i]) &&
                index->data_type[i] == (int) data_type &&
                index->id_group[i] == (int) id_group &&
                index->id_tag[i] == (int) id_tag &&
                index->user_index[i] == user_index)
        {
            /* Match found, so break. */
            break;
        }
    }

    /* Check if tag is not present. */
    if (i == index->num_tags)
    {
        *status = OSKAR_ERR_BINARY_TAG_NOT_FOUND;
        return;
    }

    /* Get the tag data. */
    if (data_size)
        *data_size = index->data_size_bytes[i];
    if (data_offset)
        *data_offset = index->data_offset_bytes[i];
}

void oskar_binary_query_ext(const oskar_Binary* index,
        unsigned char data_type, const char* name_group, const char* name_tag,
        int user_index, size_t* block_size, size_t* data_size,
        long* data_offset, int* status)
{
    int i, lgroup, ltag;

    /* Check all inputs. */
    if (!index || !name_group || !name_tag || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check that string lengths are within range. */
    lgroup = 1 + strlen(name_group);
    ltag = 1 + strlen(name_tag);
    if (lgroup > 255 || ltag > 255)
    {
        *status = OSKAR_ERR_BINARY_TAG_TOO_LONG;
        return;
    }

    /* Find the tag in the index. */
    for (i = 0; i < index->num_tags; ++i)
    {
        if (index->extended[i] &&
                index->data_type[i] == (int) data_type &&
                index->id_group[i] == (int) lgroup &&
                index->id_tag[i] == (int) ltag &&
                index->user_index[i] == user_index)
        {
            /* Possible match: check names. */
            if (strcmp(name_group, index->name_group[i]))
                continue;
            if (strcmp(name_tag, index->name_tag[i]))
                continue;

            /* Match found, so break. */
            break;
        }
    }

    /* Check if tag is not present. */
    if (i == index->num_tags)
    {
        *status = OSKAR_ERR_BINARY_TAG_NOT_FOUND;
        return;
    }

    /* Get the tag data. */
    if (block_size)
        *block_size = index->block_size_bytes[i];
    if (data_size)
        *data_size = index->data_size_bytes[i];
    if (data_offset)
        *data_offset = index->data_offset_bytes[i];
}

#ifdef __cplusplus
}
#endif
