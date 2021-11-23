/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_copy(oskar_Mem* dst, const oskar_Mem* src, int* status)
{
    if (*status || !src || !dst) return;

    /* Check the data types. */
    if (oskar_mem_type(src) != oskar_mem_type(dst))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Check the data dimensions, and resize if required. */
    if (oskar_mem_length(src) != oskar_mem_length(dst))
    {
        oskar_mem_realloc(dst, oskar_mem_length(src), status);
    }

    /* Copy the memory. */
    oskar_mem_copy_contents(dst, src, 0, 0, oskar_mem_length(src), status);
}

#ifdef __cplusplus
}
#endif
