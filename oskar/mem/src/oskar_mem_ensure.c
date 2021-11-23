/*
 * Copyright (c) 2019-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_ensure(oskar_Mem* mem, size_t num_elements, int* status)
{
    if (oskar_mem_length(mem) < num_elements)
    {
        oskar_mem_realloc(mem, num_elements, status);
    }
}

#ifdef __cplusplus
}
#endif
