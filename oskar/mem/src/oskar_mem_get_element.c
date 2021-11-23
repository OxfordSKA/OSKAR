/*
 * Copyright (c) 2014-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifdef OSKAR_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"
#include "utility/oskar_device.h"

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

double oskar_mem_get_element(const oskar_Mem* mem, size_t index, int* status)
{
    if (*status) return 0.0;
    const int location = mem->location;
    if (location == OSKAR_CPU)
    {
        switch (mem->type)
        {
        case OSKAR_DOUBLE:
            return ((const double*)mem->data)[index];
        case OSKAR_SINGLE:
            return ((const float*)mem->data)[index];
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        const size_t bytes = oskar_mem_element_size(mem->type);
        const void* src = ((const char*)mem->data) + bytes * index;
        switch (mem->type)
        {
        case OSKAR_DOUBLE:
        {
            double val = 0.0;
            cudaMemcpy(&val, src, bytes, cudaMemcpyDeviceToHost);
            return val;
        }
        case OSKAR_SINGLE:
        {
            float val = 0.0f;
            cudaMemcpy(&val, src, bytes, cudaMemcpyDeviceToHost);
            return (double)val;
        }
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location & OSKAR_CL)
    {
#ifdef OSKAR_HAVE_OPENCL
        /* Really, don't do this. There's a better way. */
        const size_t bytes = oskar_mem_element_size(mem->type);
        const size_t offset = bytes * index;
        switch (mem->type)
        {
        case OSKAR_DOUBLE:
        {
            double val = 0.0;
            clEnqueueReadBuffer(oskar_device_queue_cl(),
                    mem->buffer, CL_TRUE, offset, bytes, &val, 0, NULL, NULL);
            return val;
        }
        case OSKAR_SINGLE:
        {
            float val = 0.0f;
            clEnqueueReadBuffer(oskar_device_queue_cl(),
                    mem->buffer, CL_TRUE, offset, bytes, &val, 0, NULL, NULL);
            return (double)val;
        }
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
#else
        *status = OSKAR_ERR_OPENCL_NOT_AVAILABLE;
#endif
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
    return 0.0;
}

#ifdef __cplusplus
}
#endif
