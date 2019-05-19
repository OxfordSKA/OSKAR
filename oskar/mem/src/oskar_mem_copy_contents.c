/*
 * Copyright (c) 2011-2019, The University of Oxford
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
#include "utility/oskar_device.h"

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_copy_contents(oskar_Mem* dst, const oskar_Mem* src,
        size_t offset_dst, size_t offset_src, size_t num_elements, int* status)
{
    void *destination;
    if (*status) return;
    if (src->num_elements == 0 || num_elements == 0)
        return;
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
        cl_int error;
        error = clEnqueueWriteBuffer(oskar_device_queue_cl(),
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
        cl_int error;
        error = clEnqueueReadBuffer(oskar_device_queue_cl(),
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
        cl_int error;
        error = clEnqueueCopyBuffer(oskar_device_queue_cl(),
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
        *status = OSKAR_ERR_BAD_LOCATION;
}

#ifdef __cplusplus
}
#endif
