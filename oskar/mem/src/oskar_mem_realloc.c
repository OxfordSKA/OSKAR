/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifdef OSKAR_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"
#include "utility/oskar_device.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_realloc(oskar_Mem* mem, size_t num_elements, int* status)
{
    if (*status || !mem) return;

    /* Check if the structure owns the memory it points to. */
    if (mem->owner == 0)
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }

    /* Get size of new and old memory blocks. */
    const size_t element_size = oskar_mem_element_size(mem->type);
    if (element_size == 0)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    const size_t new_size = num_elements * element_size;
    const size_t old_size = mem->num_elements * element_size;

    /* Do nothing if new size and old size are the same. */
    if (new_size == old_size) return;

    /* Check memory location. */
    if (mem->location == OSKAR_CPU)
    {
        /* Reallocate the memory. */
        void* mem_new = realloc(mem->data, new_size);
        if (!mem_new && (new_size > 0))
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            return;
        }

        /* Initialise the new memory if it's larger than the old block. */
        if (new_size > old_size)
        {
            memset((char*)mem_new + old_size, 0, new_size - old_size);
        }

        /* Set the new meta-data. */
        mem->data = (new_size > 0) ? mem_new : 0;
        mem->num_elements = num_elements;
    }
    else if (mem->location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        /* Allocate and initialise a new block of memory. */
        void* mem_new = NULL;
        if (new_size > 0)
        {
            const int cuda_error = (int)cudaMalloc(&mem_new, new_size);
            if (cuda_error)
            {
                *status = cuda_error;
                return;
            }
            if (!mem_new)
            {
                *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
                return;
            }
        }

        /* Copy contents of old block to new block. */
        const size_t copy_size = (old_size > new_size) ? new_size : old_size;
        if (copy_size > 0)
        {
            const int cuda_error = (int)cudaMemcpy(mem_new,
                    mem->data, copy_size, cudaMemcpyDeviceToDevice);
            if (cuda_error)
            {
                *status = cuda_error;
            }
        }

        /* Free the old block. */
        const int cuda_error = (int)cudaFree(mem->data);
        if (cuda_error && !*status) *status = cuda_error;

        /* Set the new meta-data. */
        mem->data = mem_new;
        mem->num_elements = num_elements;
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (mem->location & OSKAR_CL)
    {
#ifdef OSKAR_HAVE_OPENCL
        /* Allocate and initialise a new block of memory. */
        cl_event event = 0;
        cl_int error = 0;
        cl_mem mem_new = 0;
        mem_new = clCreateBuffer(oskar_device_context_cl(),
                CL_MEM_READ_WRITE, new_size, NULL, &error);
        if (error != CL_SUCCESS)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            return;
        }

        /* Copy contents of old block to new block. */
        const size_t copy_size = (old_size > new_size) ? new_size : old_size;
        if (copy_size > 0)
        {
            error = clEnqueueCopyBuffer(oskar_device_queue_cl(), mem->buffer,
                    mem_new, 0, 0, copy_size, 0, NULL, &event);
            if (error != CL_SUCCESS)
            {
                *status = OSKAR_ERR_MEMORY_COPY_FAILURE;
            }
        }

        /* Free the old buffer. */
        if (mem->buffer)
        {
            clReleaseMemObject(mem->buffer);
        }

        /* Set the new meta-data. */
        mem->buffer = mem_new;
        mem->data = (void*) (mem->buffer);
        mem->num_elements = num_elements;
#else
        *status = OSKAR_ERR_OPENCL_NOT_AVAILABLE;
#endif
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
}

#ifdef __cplusplus
}
#endif
