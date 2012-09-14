/*
 * Copyright (c) 2012, The University of Oxford
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

#include "utility/oskar_mem_alloc.h"
#include "utility/oskar_mem_element_size.h"
#include <cuda_runtime_api.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_alloc(oskar_Mem* mem, int* status)
{
    int n_elements, location, type;
    size_t element_size, bytes;

    /* Check all inputs. */
    if (!mem || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check if the structure owns the memory it points to. */
    if (!mem->owner)
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }

    /* Get the meta-data. */
    n_elements = mem->num_elements;
    location = mem->location;
    type = mem->type;

    /* Check if allocation should happen or not. */
    if (n_elements == 0)
        return;

    /* Get the memory size. */
    element_size = oskar_mem_element_size(type);
    if (element_size == 0)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    bytes = n_elements * element_size;

    /* Check whether the memory should be on the host or the device. */
    if (location == OSKAR_LOCATION_CPU)
    {
        /* Allocate host memory. */
        mem->data = calloc(bytes, 1);
        if (mem->data == NULL)
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
    }
    else if (location == OSKAR_LOCATION_GPU)
    {
        /* Allocate GPU memory. */
        cudaMalloc(&mem->data, bytes);
        cudaMemset(mem->data, 0, bytes);
        *status = cudaPeekAtLastError();
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
}

#ifdef __cplusplus
}
#endif
