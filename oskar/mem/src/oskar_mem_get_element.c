/*
 * Copyright (c) 2014-2019, The University of Oxford
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
            double val;
            cudaMemcpy(&val, src, bytes, cudaMemcpyDeviceToHost);
            return val;
        }
        case OSKAR_SINGLE:
        {
            float val;
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
            double val;
            clEnqueueReadBuffer(oskar_device_queue_cl(),
                    mem->buffer, CL_TRUE, offset, bytes, &val, 0, NULL, NULL);
            return val;
        }
        case OSKAR_SINGLE:
        {
            float val;
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
        *status = OSKAR_ERR_BAD_LOCATION;
    return 0.0;
}

#ifdef __cplusplus
}
#endif
