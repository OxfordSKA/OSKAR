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

#include "log/oskar_log.h"
#include "mem/oskar_mem.h"
#include "mem/private_mem.h"
#include "utility/oskar_device.h"

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_clear_contents(oskar_Mem* mem, int* status)
{
    if (*status || mem->num_elements == 0) return;
    const size_t size = mem->num_elements * oskar_mem_element_size(mem->type);
    if (mem->location == OSKAR_CPU)
        memset(mem->data, 0, size);
    else if (mem->location == OSKAR_GPU)
#ifdef OSKAR_HAVE_CUDA
        cudaMemset(mem->data, 0, size);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    else if (mem->location & OSKAR_CL)
    {
#ifdef OSKAR_HAVE_OPENCL
        cl_event event;
        cl_int error;
        char zero = '\0';
        error = clEnqueueFillBuffer(oskar_device_queue_cl(),
                mem->buffer, &zero, sizeof(char), 0, size, 0, NULL, &event);
        clWaitForEvents(1, &event); /* This is required. */
        if (error != CL_SUCCESS)
        {
            oskar_log_error("clEnqueueFillBuffer() error (%d)", error);
            *status = OSKAR_ERR_INVALID_ARGUMENT;
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
