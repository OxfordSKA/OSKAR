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

#include "utility/oskar_mem_append.h"
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_mem_element_size.h"
#include "utility/oskar_Mem.h"

#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


int oskar_mem_append(oskar_Mem* to, const void* from, int type, int location,
        int num_elements)
{
    // Check for sane inputs.
    if (to == NULL || from == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    // Check that the data types match.
    if (to->private_type != type)
        return OSKAR_ERR_TYPE_MISMATCH;

    // Memory size being appended and offset into memory to append to.
    size_t element_size = oskar_mem_element_size(to->private_type);
    size_t mem_size = num_elements * element_size;
    size_t offset_bytes = to->private_n_elements * element_size;

    // Reallocate the memory pointer so it is big enough to append to.
    int error = 0;
    error = oskar_mem_realloc(to, num_elements + to->private_n_elements);
    if (error != 0) return error;

    // Append to the memory.
    if (location == OSKAR_LOCATION_CPU)
    {
        if (to->private_location == OSKAR_LOCATION_CPU)
            memcpy((char*)(to->data) + offset_bytes, from, mem_size);
        else
            cudaMemcpy((char*)(to->data) + offset_bytes, from,
                    mem_size, cudaMemcpyHostToDevice);
    }
    else if (location == OSKAR_LOCATION_GPU)
    {
        if (to->private_location == OSKAR_LOCATION_CPU)
            cudaMemcpy((char*)(to->data) + offset_bytes, from,
                    mem_size, cudaMemcpyDeviceToHost);
        else
            cudaMemcpy((char*)(to->data) + offset_bytes, from,
                    mem_size, cudaMemcpyDeviceToDevice);
    }
    else
    {
        return OSKAR_ERR_BAD_LOCATION;
    }
    return error;
}
