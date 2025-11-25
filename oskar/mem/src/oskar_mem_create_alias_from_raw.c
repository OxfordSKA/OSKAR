/*
 * Copyright (c) 2014-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"

#ifdef __cplusplus
extern "C" {
#endif


oskar_Mem* oskar_mem_create_alias_from_raw(
        void* ptr,
        int type,
        int location,
        size_t num_elements,
        int* status
)
{
    oskar_Mem* mem = (oskar_Mem*) calloc(1, sizeof(oskar_Mem));
    if (!mem)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;         /* LCOV_EXCL_LINE */
        return 0;                                         /* LCOV_EXCL_LINE */
    }
    mem->type = type;
    mem->location = location;
    mem->num_elements = num_elements;
    mem->owner = 0;
    mem->data = ptr;
    mem->ref_count = 1;
    mem->mutex = oskar_mutex_create();
    return mem;
}

#ifdef __cplusplus
}
#endif
