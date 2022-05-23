/*
 * Copyright (c) 2014-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Mem* oskar_mem_create_alias(const oskar_Mem* src, size_t offset,
        size_t num_elements, int* status)
{
    oskar_Mem* mem = (oskar_Mem*) calloc(1, sizeof(oskar_Mem));
    if (!mem)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return 0;
    }

    /* Initialise meta-data.
     * (This must happen regardless of the status code.) */
    mem->owner = 0; /* Structure does not own the memory. */
    mem->ref_count = 1;
    mem->mutex = oskar_mutex_create();
    if (src)
    {
        size_t element_size = oskar_mem_element_size(src->type);
        mem->type = src->type;
        mem->location = src->location;
        mem->num_elements = num_elements;
#ifdef OSKAR_HAVE_OPENCL
        if (mem->location & OSKAR_CL)
        {
            cl_int error = 0;
            cl_buffer_region r;
            r.origin = element_size * offset;
            r.size   = element_size * num_elements;
            mem->buffer = clCreateSubBuffer(src->buffer,
                    CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                    &r, &error);
            mem->data = (void*) (mem->buffer);
            if (error != CL_SUCCESS)
            {
                *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            }
        }
        else
#endif
        {
            size_t offset_bytes = offset * element_size;
            mem->data = (void*)(((char*)(src->data)) + offset_bytes);
        }
    }

    /* Return a handle to the new structure. */
    return mem;
}

#ifdef __cplusplus
}
#endif
