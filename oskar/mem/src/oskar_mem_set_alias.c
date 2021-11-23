/*
 * Copyright (c) 2014-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_set_alias(oskar_Mem* mem, const oskar_Mem* src, size_t offset,
        size_t num_elements, int* status)
{
    size_t element_size = 0;

    /* The destination structure must not own its memory.
     * The structure must have been created using oskar_mem_create_alias*(),
     * so the owner flag must be set to false. */
    if (mem->owner)
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }

    /* Check that the new pointer will be valid. */
    if (offset + num_elements > src->num_elements)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    /* Get the element size. */
    element_size = oskar_mem_element_size(src->type);
    if (element_size == 0)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Set meta-data. */
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
        clReleaseMemObject(mem->buffer);
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
        mem->data = (void*)((char*)(src->data) + offset_bytes);
    }
}

#ifdef __cplusplus
}
#endif
