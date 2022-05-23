/*
 * Copyright (c) 2013-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifdef OSKAR_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"
#include "utility/oskar_device.h"

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Mem* oskar_mem_create(int type, int location, size_t num_elements,
         int* status)
{
    oskar_Mem* mem = (oskar_Mem*) calloc(1, sizeof(oskar_Mem));
    if (!mem)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return 0;
    }

    /* Initialise meta-data.
     * (This must happen regardless of the status code.) */
    mem->type = type;
    mem->location = location;
    mem->num_elements = 0;
    mem->owner = 1;
    mem->data = NULL;
    mem->ref_count = 1;
    mem->mutex = oskar_mutex_create();

    /* Check if allocation should happen or not. */
    if (!status || *status || num_elements == 0)
    {
        return mem;
    }

    /* Get the memory size. */
    const size_t element_size = oskar_mem_element_size(type);
    if (element_size == 0)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return mem;
    }
    const size_t bytes = num_elements * element_size;

    /* Check whether the memory should be on the host or the device. */
    mem->num_elements = num_elements;
    if (location == OSKAR_CPU)
    {
        /* Allocate host memory. */
        mem->data = calloc(bytes, 1);
        if (mem->data == NULL)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            return mem;
        }
        /* The memset() call forces the allocation
         * to actually happen by touching the whole block.
         * This makes subsequent copies much faster. */
        memset(mem->data, 0, bytes);
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        /* Allocate GPU memory. For efficiency, don't clear it. */
        *status = (int)cudaMalloc(&mem->data, bytes);
        if (!*status && mem->data == NULL)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        }
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location & OSKAR_CL)
    {
#ifdef OSKAR_HAVE_OPENCL
        /* Allocate OpenCL memory buffer using the current context. */
        cl_int error = 0;
        mem->buffer = clCreateBuffer(oskar_device_context_cl(),
                CL_MEM_READ_WRITE, bytes, NULL, &error);
        if (error != CL_SUCCESS)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        }
        mem->data = (void*) (mem->buffer);
#else
        *status = OSKAR_ERR_OPENCL_NOT_AVAILABLE;
#endif
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }

    /* Return a handle to the structure .*/
    return mem;
}

#ifdef __cplusplus
}
#endif
