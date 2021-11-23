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

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_set_element_real(oskar_Mem* mem, size_t index,
        double val, int* status)
{
    if (*status) return;
    const int precision = oskar_type_precision(mem->type);
    const int location = mem->location;
    if (location == OSKAR_CPU)
    {
        switch (precision)
        {
        case OSKAR_DOUBLE:
            ((double*)(mem->data))[index] = val;
            return;
        case OSKAR_SINGLE:
            ((float*)(mem->data))[index] = (float) val;
            return;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        switch (precision)
        {
        case OSKAR_DOUBLE:
        {
            cudaMemcpy((double*)(mem->data) + index, &val, sizeof(double),
                    cudaMemcpyHostToDevice);
            return;
        }
        case OSKAR_SINGLE:
        {
            const float val_f = (float) val;
            cudaMemcpy((float*)(mem->data) + index, &val_f, sizeof(float),
                    cudaMemcpyHostToDevice);
            return;
        }
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location & OSKAR_CL)
    {
#ifdef OSKAR_HAVE_OPENCL
        const size_t bytes = oskar_mem_element_size(mem->type);
        const size_t offset = bytes * index;
        const float val_f = (float) val;
        const void* ptr = 0;
        if (precision == OSKAR_DOUBLE)
        {
            ptr = &val;
        }
        else if (precision == OSKAR_SINGLE)
        {
            ptr = &val_f;
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        const cl_int error = clEnqueueWriteBuffer(oskar_device_queue_cl(),
                mem->buffer, CL_TRUE, offset, bytes, ptr, 0, NULL, NULL);
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
