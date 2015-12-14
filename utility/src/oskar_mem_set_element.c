/*
 * Copyright (c) 2014-2015, The University of Oxford
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

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_set_element_scalar_real(oskar_Mem* mem, size_t index,
        double val, int* status)
{
    int type, location;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check index is within range. */
    if (index >= mem->num_elements)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    /* Get the data type and location */
    type = mem->type;
    location = mem->location;

    /* Set the data into the array element. */
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_DOUBLE)
        {
            ((double*)(mem->data))[index] = val;
            return;
        }
        else if (type == OSKAR_SINGLE)
        {
            ((float*)(mem->data))[index] = (float) val;
            return;
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_DOUBLE)
        {
            cudaMemcpy((double*)(mem->data) + index, &val, sizeof(double),
                    cudaMemcpyHostToDevice);
            return;
        }
        else if (type == OSKAR_SINGLE)
        {
            float temp;
            temp = (float) val;
            cudaMemcpy((float*)(mem->data) + index, &temp, sizeof(float),
                    cudaMemcpyHostToDevice);
            return;
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else
        *status = OSKAR_ERR_BAD_LOCATION;
}

#ifdef __cplusplus
}
#endif
