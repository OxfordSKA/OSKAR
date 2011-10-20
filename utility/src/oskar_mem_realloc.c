/*
 * Copyright (c) 2011, The University of Oxford
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

#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_mem_element_size.h"

#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>

int oskar_mem_realloc(oskar_Mem* mem, int num_elements)
{
    if (mem == NULL) return OSKAR_ERR_INVALID_ARGUMENT;

    size_t element_size = oskar_mem_element_size(mem->private_type);
    size_t new_size = num_elements * element_size;
    int error = 0;
    if (mem->private_location == OSKAR_LOCATION_CPU)
    {
        void* mem_new = realloc(mem->data, new_size);
        if (mem_new == NULL)
            return OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        mem->data = mem_new;
        mem->private_n_elements = num_elements;
    }
    else if (mem->private_location == OSKAR_LOCATION_GPU)
    {
        size_t old_size = mem->private_n_elements * element_size;
        void* d_mem_new = NULL;
        cudaMalloc(&d_mem_new, new_size);
        cudaMemcpy(d_mem_new, mem->data, old_size, cudaMemcpyDeviceToDevice);
        cudaFree(mem->data);
        mem->data = d_mem_new;
        error = cudaPeekAtLastError();
        mem->private_n_elements = num_elements;
    }
    else
    {
        return OSKAR_ERR_UNKNOWN;
    }
    return error;
}
