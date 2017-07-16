/*
 * Copyright (c) 2011-2017, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifdef OSKAR_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"
#include "utility/oskar_cl_utils.h"
#include "utility/oskar_device_utils.h"

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_copy_contents(oskar_Mem* dst, const oskar_Mem* src,
        size_t offset_dst, size_t offset_src, size_t num_elements, int* status)
{
    int location_src, location_dst;
    size_t bytes, element_size, start_dst, start_src;
    void *destination;
    const void *source;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Return immediately if there is nothing to copy. */
    if (src->num_elements == 0 || num_elements == 0)
        return;

    /* Check the data types. */
    if (src->type != dst->type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Check the data dimensions. */
    if (num_elements > src->num_elements ||
            num_elements > (dst->num_elements - offset_dst))
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    /* Get the number of bytes to copy. */
    element_size = oskar_mem_element_size(src->type);
    bytes        = element_size * num_elements;
    start_dst    = element_size * offset_dst;
    start_src    = element_size * offset_src;
    destination  = (void*)((char*)(dst->data) + start_dst);
    source       = (const void*)((const char*)(src->data) + start_src);
    location_src = src->location;
    location_dst = dst->location;

    /* Host to host. */
    if (location_src == OSKAR_CPU && location_dst == OSKAR_CPU)
    {
        memcpy(destination, source, bytes);
        return;
    }

    /* Host to CUDA device. */
    else if (location_src == OSKAR_CPU && location_dst == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice);
        oskar_device_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        return;
    }

    /* CUDA device to host. */
    else if (location_src == OSKAR_GPU && location_dst == OSKAR_CPU)
    {
#ifdef OSKAR_HAVE_CUDA
        cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost);
        oskar_device_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        return;
    }

    /* CUDA device to CUDA device. */
    else if (location_src == OSKAR_GPU && location_dst == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice);
        oskar_device_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        return;
    }

    /* Host to OpenCL device. */
    else if (location_src == OSKAR_CPU && (location_dst & OSKAR_CL))
    {
#ifdef OSKAR_HAVE_OPENCL
        cl_int error;
        if (!source)
        {
            *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
            return;
        }
        error = clEnqueueWriteBuffer(oskar_cl_command_queue(), dst->buffer,
                CL_TRUE, start_dst, bytes, source, 0, NULL, NULL);
        if (error != CL_SUCCESS)
        {
            size_t available;
            clGetMemObjectInfo(dst->buffer, CL_MEM_SIZE,
                    sizeof(size_t), &available, NULL);
            fprintf(stderr, "clEnqueueWriteBuffer() error (%d): "
                    "Copy size: %lu, Offset: %lu, Available: %lu\n",
                    error,
                    (unsigned long) bytes,
                    (unsigned long) start_dst,
                    (unsigned long) available);
            *status = OSKAR_ERR_MEMORY_COPY_FAILURE;
        }
#else
        *status = OSKAR_ERR_OPENCL_NOT_AVAILABLE;
#endif
        return;
    }

    /* OpenCL device to host. */
    else if ((location_src & OSKAR_CL) && location_dst == OSKAR_CPU)
    {
#ifdef OSKAR_HAVE_OPENCL
        cl_int error;
        error = clEnqueueReadBuffer(oskar_cl_command_queue(), src->buffer,
                CL_TRUE, start_src, bytes, destination, 0, NULL, NULL);
        if (error != CL_SUCCESS)
        {
            fprintf(stderr, "clEnqueueReadBuffer() error (%d)\n", error);
            *status = OSKAR_ERR_MEMORY_COPY_FAILURE;
        }
#else
        *status = OSKAR_ERR_OPENCL_NOT_AVAILABLE;
#endif
        return;
    }

    /* OpenCL device to OpenCL device. */
    else if ((location_src & OSKAR_CL) && (location_dst & OSKAR_CL))
    {
#ifdef OSKAR_HAVE_OPENCL
        cl_int error;
        error = clEnqueueCopyBuffer(oskar_cl_command_queue(), src->buffer,
                dst->buffer, start_src, start_dst, bytes, 0, NULL, NULL);
        if (error != CL_SUCCESS)
        {
            fprintf(stderr, "clEnqueueCopyBuffer() error (%d)\n", error);
            *status = OSKAR_ERR_MEMORY_COPY_FAILURE;
        }
#else
        *status = OSKAR_ERR_OPENCL_NOT_AVAILABLE;
#endif
        return;
    }

    *status = OSKAR_ERR_BAD_LOCATION;
}

#ifdef __cplusplus
}
#endif
