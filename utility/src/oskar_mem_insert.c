/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_insert(oskar_Mem* dst, const oskar_Mem* src, int offset,
        int* status)
{
    int n_elements_src, n_elements_dst, type_src, type_dst;
    int location_src, location_dst;
    size_t bytes, start;
    void* destination;

    /* Check all inputs. */
    if (!src || !dst || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the meta-data. */
    n_elements_src = src->num_elements;
    n_elements_dst = dst->num_elements;
    type_src = src->type;
    type_dst = dst->type;
    location_src = src->location;
    location_dst = dst->location;

    /* Return immediately if there is nothing to copy. */
    if (src->data == NULL || n_elements_src == 0)
        return;

    /* Check the data types. */
    if (type_src != type_dst)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Check the data dimensions. */
    if (n_elements_src > (n_elements_dst - offset))
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    /* Get the number of bytes to copy. */
    bytes = oskar_mem_element_size(type_src) * n_elements_src;
    start = oskar_mem_element_size(type_src) * offset;
    destination = (void*)((char*)(dst->data) + start);

    /* Host to host. */
    if (location_src == OSKAR_LOCATION_CPU
            && location_dst == OSKAR_LOCATION_CPU)
    {
        memcpy(destination, src->data, bytes);
        return;
    }

    /* Host to device. */
    else if (location_src == OSKAR_LOCATION_CPU
            && location_dst == OSKAR_LOCATION_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        cudaMemcpy(destination, src->data, bytes, cudaMemcpyHostToDevice);
        *status = cudaPeekAtLastError();
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        return;
    }

    /* Device to host. */
    else if (location_src == OSKAR_LOCATION_GPU
            && location_dst == OSKAR_LOCATION_CPU)
    {
#ifdef OSKAR_HAVE_CUDA
        cudaMemcpy(destination, src->data, bytes, cudaMemcpyDeviceToHost);
        *status = cudaPeekAtLastError();
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        return;
    }

    /* Device to device. */
    else if (location_src == OSKAR_LOCATION_GPU
            && location_dst == OSKAR_LOCATION_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        cudaMemcpy(destination, src->data, bytes, cudaMemcpyDeviceToDevice);
        *status = cudaPeekAtLastError();
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        return;
    }

    *status = OSKAR_ERR_BAD_LOCATION;
}

#ifdef __cplusplus
}
#endif
