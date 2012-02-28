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
#include "utility/oskar_binary_tag_index_free.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_binary_tag_index_free(oskar_BinaryTagIndex** index)
{
    int i;

    /* Sanity check on inputs. */
    if (index == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Check if index needs to be freed. */
    if (*index == NULL)
        return OSKAR_SUCCESS;

    /* Free string data. */
    for (i = 0; i < (*index)->num_tags; ++i)
    {
        free((*index)->name_group[i]);
        free((*index)->name_tag[i]);
    }

    /* Free arrays. */
    free((*index)->extended);
    free((*index)->data_type);
    free((*index)->id_group);
    free((*index)->id_tag);
    free((*index)->name_group);
    free((*index)->name_tag);
    free((*index)->user_index);
    free((*index)->data_offset_bytes);
    free((*index)->data_size_bytes);
    free((*index)->block_size_bytes);

    /* Free the structure itself. */
    free(*index);
    *index = NULL;

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
