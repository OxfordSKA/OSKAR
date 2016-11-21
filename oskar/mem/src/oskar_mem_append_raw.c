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

#include <private_mem.h>
#include <oskar_mem.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_append_raw(oskar_Mem* to, const void* from, int from_type,
        int from_location, size_t num_elements, int* status)
{
    size_t element_size, mem_size, offset_bytes;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check that the data types match. */
    if (to->type != from_type)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Memory size being appended and offset into memory to append to. */
    element_size = oskar_mem_element_size(to->type);
    mem_size = num_elements * element_size;
    offset_bytes = to->num_elements * element_size;

    /* Reallocate the memory block so it is big enough to hold the new data. */
    oskar_mem_realloc(to, num_elements + to->num_elements, status);

    /* Check if safe to proceed. */
    if (*status) return;

    /* Append to the memory. */
    if (from_location == OSKAR_CPU)
    {
        if (to->location == OSKAR_CPU)
            memcpy((char*)(to->data) + offset_bytes, from, mem_size);
        else
        {
#ifdef OSKAR_HAVE_CUDA
            cudaMemcpy((char*)(to->data) + offset_bytes, from,
                    mem_size, cudaMemcpyHostToDevice);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
    }
    else if (from_location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (to->location == OSKAR_CPU)
            cudaMemcpy((char*)(to->data) + offset_bytes, from,
                    mem_size, cudaMemcpyDeviceToHost);
        else
            cudaMemcpy((char*)(to->data) + offset_bytes, from,
                    mem_size, cudaMemcpyDeviceToDevice);
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
