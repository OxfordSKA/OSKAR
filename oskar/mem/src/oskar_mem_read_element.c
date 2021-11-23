/*
 * Copyright (c) 2019-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifdef OSKAR_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"
#include "utility/oskar_device.h"

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_read_element(const oskar_Mem* mem, size_t index,
        void* out, int* status)
{
    if (*status) return;
    const size_t bytes = oskar_mem_element_size(mem->type);
    const size_t offset = bytes * index;
    if (mem->location == OSKAR_CPU)
    {
        const char* from = ((const char*) mem->data) + offset;
        memcpy(out, (const void*)from, bytes);
    }
    else if (mem->location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        const char* from = ((const char*) mem->data) + offset;
        cudaMemcpy(out, (const void*)from, bytes, cudaMemcpyDeviceToHost);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (mem->location & OSKAR_CL)
    {
#ifdef OSKAR_HAVE_OPENCL
        const cl_int error = clEnqueueReadBuffer(oskar_device_queue_cl(),
                mem->buffer, CL_TRUE, offset, bytes, out, 0, NULL, NULL);
        if (error != CL_SUCCESS)
        {
            *status = OSKAR_ERR_MEMORY_COPY_FAILURE;
        }
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
