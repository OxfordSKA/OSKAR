/*
 * Copyright (c) 2013-2017, The University of Oxford
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

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Mem* oskar_mem_create(int type, int location, size_t num_elements,
         int* status)
{
    oskar_Mem* mem = 0;
    size_t element_size, bytes;

    /* Create the structure. */
    mem = (oskar_Mem*) malloc(sizeof(oskar_Mem));
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

    /* Check if allocation should happen or not. */
    if (!status || *status || num_elements == 0)
        return mem;

    /* Get the memory size. */
    element_size = oskar_mem_element_size(type);
    if (element_size == 0)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return mem;
    }
    bytes = num_elements * element_size;

    /* Check whether the memory should be on the host or the device. */
    mem->num_elements = num_elements;
    if (location == OSKAR_CPU)
    {
        /* Allocate host memory. */
        mem->data = calloc(bytes, 1);
        if (mem->data == NULL)
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        /* Allocate GPU memory. For efficiency, don't clear it. */
        cudaMalloc(&mem->data, bytes);
        if (mem->data == NULL)
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        oskar_device_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location & OSKAR_CL)
    {
#ifdef OSKAR_HAVE_OPENCL
        /* Allocate OpenCL memory buffer using the current context. */
        cl_int error = 0;
        mem->buffer = clCreateBuffer(oskar_cl_context(),
                CL_MEM_READ_WRITE, bytes, NULL, &error);
        if (error != CL_SUCCESS)
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
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
