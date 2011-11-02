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

#include "utility/oskar_mem_element_size.h"
#include "utility/oskar_mem_realloc.h"

#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_mem_realloc(oskar_Mem* mem, int num_elements)
{
    size_t element_size, new_size, old_size;
    int error = 0;

    /* Check for sane inputs. */
    if (mem == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Get size of new and old memory blocks. */
    element_size = oskar_mem_element_size(mem->private_type);
    if (element_size == 0)
        return OSKAR_ERR_BAD_DATA_TYPE;
    new_size = num_elements * element_size;
    old_size = mem->private_n_elements * element_size;

    /* Check memory location. */
    if (mem->private_location == OSKAR_LOCATION_CPU)
    {
        /* Reallocate the memory. */
        void* mem_new = NULL;
        mem_new = realloc(mem->data, new_size);
        if (mem_new == NULL)
            return OSKAR_ERR_MEMORY_ALLOC_FAILURE;

        /* Set the new meta-data. */
        mem->data = mem_new;
        mem->private_n_elements = num_elements;
    }
    else if (mem->private_location == OSKAR_LOCATION_GPU)
    {
        /* Allocate a new block of memory. */
        size_t copy_size;
        void* mem_new = NULL;
        cudaMalloc(&mem_new, new_size);

        /* Copy contents of old block to new block. */
        copy_size = (old_size > new_size) ? new_size : old_size;
        cudaMemcpy(mem_new, mem->data, copy_size, cudaMemcpyDeviceToDevice);

        /* Free the old block. */
        cudaFree(mem->data);
        error = cudaPeekAtLastError();

        /* Set the new meta-data. */
        mem->data = mem_new;
        mem->private_n_elements = num_elements;
    }
    else
    {
        return OSKAR_ERR_BAD_LOCATION;
    }
    return error;
}

#ifdef __cplusplus
}
#endif
