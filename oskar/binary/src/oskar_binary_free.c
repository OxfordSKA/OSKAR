/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "binary/oskar_binary.h"
#include "binary/private_binary.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_binary_free(oskar_Binary* handle)
{
    int i = 0;
    if (!handle) return;

    /* Close the file. */
    if (handle->stream)
    {
        fclose(handle->stream);
    }

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
