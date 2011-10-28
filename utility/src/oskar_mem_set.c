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

#include "utility/oskar_mem_set.h"
#include "utility/oskar_mem_element_size.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_alloc.h"
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_Mem.h"

#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>

int oskar_mem_set(oskar_Mem* dst, void* src, int src_type, int src_num_elements,
        int src_location)
{
    int error = 0;
    size_t dst_size, src_size;

    // Check for sane inputs.
    if (dst == NULL || src == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    // Compute the sizes.
    dst_size = dst->private_n_elements *
            oskar_mem_element_size(dst->private_type);
    src_size = src_num_elements * oskar_mem_element_size(src_type);

    // If the memory size changes, free and reallocate the memory.
    if (dst_size != src_size)
    {
        int location = dst->private_location;
        error = oskar_mem_free(dst);
        if (error != 0) return error;
        dst->private_location   = location;
        dst->private_n_elements = src_num_elements;
        dst->private_type       = src_type;
        error = oskar_mem_alloc(dst);
        if (error != 0) return error;
    }

    if (src_location == OSKAR_LOCATION_CPU)
    {
        if (dst->private_location == OSKAR_LOCATION_CPU)
            memcpy(dst->data, src, src_size);
        else if (dst->private_location == OSKAR_LOCATION_GPU)
            cudaMemcpy(dst->data, src, src_size, cudaMemcpyHostToDevice);
        else
            return OSKAR_ERR_BAD_LOCATION;
    }
    else if (src_location == OSKAR_LOCATION_GPU)
    {
        if (dst->private_location == OSKAR_LOCATION_CPU)
            cudaMemcpy(dst->data, src, cudaMemcpyDeviceToHost);
        else if (dst->private_location == OSKAR_LOCATION_GPU)
            cudaMemcpy(dst->data, src, cudaMemcpyDeviceToDevice);
        else
            return OSKAR_ERR_BAD_LOCATION;
    }
    else
    {
        return OSKAR_ERR_BAD_LOCATION;
    }

    return error;
}
