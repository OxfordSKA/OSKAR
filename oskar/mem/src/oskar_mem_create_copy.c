/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

oskar_Mem* oskar_mem_create_copy(const oskar_Mem* src, int location,
        int* status)
{
    oskar_Mem* mem = 0;
    if (*status) return 0;

    /* Create the new structure. */
    mem = oskar_mem_create(oskar_mem_type(src), location,
            oskar_mem_length(src), status);
    if (!mem || *status)
    {
        return mem;
    }

    /* Copy the memory. */
    oskar_mem_copy_contents(mem, src, 0, 0, oskar_mem_length(src), status);

    /* Return a handle to the new structure. */
    return mem;
}

#ifdef __cplusplus
}
#endif
