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

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_copy_contents(oskar_Mem* dst, const oskar_Mem* src,
        size_t offset_dst, size_t offset_src, size_t num_elements, int* status)
{
    void *destination = 0;
    if (*status) return;
    if (src->num_elements == 0 || num_elements == 0)
    {
        return;
    }
    if (src->type != dst->type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (num_elements > src->num_elements ||
            num_elements > (dst->num_elements - offset_dst))
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }
    const size_t element_size = oskar_mem_element_size(src->type);
    const size_t bytes        = element_size * num_elements;
    const size_t start_dst    = element_size * offset_dst;
    const size_t start_src    = element_size * offset_src;
    const int location_src    = src->location;
    const int location_dst    = dst->location;
    const void *source = (const void*)((const char*)(src->data) + start_src);
    destination        = (void*)((char*)(dst->data) + start_dst);

    /* Host to host. */
    if (location_src == OSKAR_CPU && location_dst == OSKAR_CPU)
    {
        memcpy(destination, source, bytes);
    }

    /* Host to CUDA device. */
    else if (location_src == OSKAR_CPU && location_dst == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        *status = (int)cudaMemcpy(destination, source, bytes,
                cudaMemcpyHostToDevice);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }

    /* CUDA device to host. */
    else if (location_src == OSKAR_GPU && location_dst == OSKAR_CPU)
    {
#ifdef OSKAR_HAVE_CUDA
        *status = (int)cudaMemcpy(destination, source, bytes,
                cudaMemcpyDeviceToHost);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }

    /* CUDA device to CUDA device. */
    else if (location_src == OSKAR_GPU && location_dst == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        *status = (int)cudaMemcpy(destination, source, bytes,
                cudaMemcpyDeviceToDevice);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }

    /* Host to OpenCL device. */
    else if (location_src == OSKAR_CPU && (location_dst & OSKAR_CL))
    {
#ifdef OSKAR_HAVE_OPENCL
        const cl_int error = clEnqueueWriteBuffer(oskar_device_queue_cl(),
                dst->buffer, CL_TRUE, start_dst, bytes, source,
                0, NULL, NULL);
        if (error != CL_SUCCESS)
        {
            fprintf(stderr, "clEnqueueWriteBuffer() error (%d)\n", error);
            *status = OSKAR_ERR_MEMORY_COPY_FAILURE;
        }
#else
        *status = OSKAR_ERR_OPENCL_NOT_AVAILABLE;
#endif
    }

    /* OpenCL device to host. */
    else if ((location_src & OSKAR_CL) && location_dst == OSKAR_CPU)
    {
#ifdef OSKAR_HAVE_OPENCL
        const cl_int error = clEnqueueReadBuffer(oskar_device_queue_cl(),
                src->buffer, CL_TRUE, start_src, bytes, destination,
                0, NULL, NULL);
        if (error != CL_SUCCESS)
        {
            fprintf(stderr, "clEnqueueReadBuffer() error (%d)\n", error);
            *status = OSKAR_ERR_MEMORY_COPY_FAILURE;
        }
#else
        *status = OSKAR_ERR_OPENCL_NOT_AVAILABLE;
#endif
    }

    /* OpenCL device to OpenCL device. */
    else if ((location_src & OSKAR_CL) && (location_dst & OSKAR_CL))
    {
#ifdef OSKAR_HAVE_OPENCL
        const cl_int error = clEnqueueCopyBuffer(oskar_device_queue_cl(),
                src->buffer, dst->buffer, start_src, start_dst, bytes,
                0, NULL, NULL);
        if (error != CL_SUCCESS)
        {
            fprintf(stderr, "clEnqueueCopyBuffer() error (%d)\n", error);
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
