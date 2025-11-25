/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"

#ifdef __cplusplus
extern "C" {
#endif


void oskar_mem_append_raw(
        oskar_Mem* to,
        const void* from,
        int from_type,
        int from_location,
        size_t num_elements,
        int* status
)
{
    if (*status) return;

    /* Check that the data types match. */
    if (to->type != from_type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;                /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }

    /* Check that the locations are supported by the current function. */
    if (to->location != OSKAR_CPU || from_location != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;                 /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }

    /* Get source memory size and destination offset. */
    const size_t element_size = oskar_mem_element_size(to->type);
    const size_t mem_size = num_elements * element_size;
    const size_t offset_bytes = to->num_elements * element_size;

    /* Reallocate the memory block so it is big enough to hold the new data. */
    oskar_mem_realloc(to, num_elements + to->num_elements, status);
    if (*status) return;

    /* Append to the memory. */
    if (from_location == OSKAR_CPU && to->location == OSKAR_CPU)
    {
        memcpy((char*) (to->data) + offset_bytes, from, mem_size);
    }
}

#ifdef __cplusplus
}
#endif
