/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include <oskar_cuda_check_error.h>
#include <oskar_mem.h>
#include <private_mem.h>

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_realloc(oskar_Mem* mem, size_t num_elements, int* status)
{
    size_t element_size, new_size, old_size;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check if the structure owns the memory it points to. */
    if (mem->owner == 0)
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }

    /* Get size of new and old memory blocks. */
    element_size = oskar_mem_element_size(mem->type);
    if (element_size == 0)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    new_size = num_elements * element_size;
    old_size = mem->num_elements * element_size;

    /* Do nothing if new size and old size are the same. */
    if (new_size == old_size)
        return;

    /* Check memory location. */
    if (mem->location == OSKAR_CPU)
    {
        /* Reallocate the memory. */
        void* mem_new = NULL;
        mem_new = realloc(mem->data, new_size);
        if (!mem_new)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            return;
        }

        /* Initialise the new memory if it's larger than the old block. */
        if (new_size > old_size)
            memset((char*)mem_new + old_size, 0, new_size - old_size);

        /* Set the new meta-data. */
        mem->data = mem_new;
        mem->num_elements = num_elements;
    }
    else if (mem->location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        /* Allocate and initialise a new block of memory. */
        int cuda_error = 0;
        size_t copy_size;
        void* mem_new = NULL;
        cuda_error = cudaMalloc(&mem_new, new_size);
        if (cuda_error)
        {
            *status = cuda_error;
            return;
        }
        if (!mem_new)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            return;
        }

        /* Copy contents of old block to new block. */
        copy_size = (old_size > new_size) ? new_size : old_size;
        cuda_error = cudaMemcpy(mem_new, mem->data, copy_size,
                cudaMemcpyDeviceToDevice);
        if (cuda_error)
        {
            *status = cuda_error;
            return;
        }

        /* Free the old block. */
        cudaFree(mem->data);
        oskar_cuda_check_error(status);

        /* Set the new meta-data. */
        mem->data = mem_new;
        mem->num_elements = num_elements;
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
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
