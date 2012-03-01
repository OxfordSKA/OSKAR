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

#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_Mem.h"

// Single precision kernel.
__global__
static void oskar_cudak_scale_real_f(int n, float s, float* a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        a[i] *= s;
}

// Double precision kernel.
__global__
static void oskar_cudak_scale_real_d(int n, double s, double* a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        a[i] *= s;
}

#ifdef __cplusplus
extern "C"
#endif
int oskar_mem_scale_real(oskar_Mem* mem, double value)
{
    int type, location, num_elements, i, num_threads, num_blocks, error = 0;

    /* Check for sane inputs. */
    if (mem == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Get memory meta-data. */
#ifdef __cplusplus
    type = mem->type();
    location = mem->location();
    num_elements = mem->num_elements();
#else
    type = mem->private_type;
    location = mem->private_location;
    num_elements = mem->private_num_elements;
#endif

    /* Check if elements are real, complex or matrix. */
    if (oskar_mem_is_complex(type))
        num_elements *= 2;
    if (oskar_mem_is_matrix(type))
        num_elements *= 4;

    /* Scale the vector. */
    if (oskar_mem_is_single(type))
    {
        if (location == OSKAR_LOCATION_CPU)
        {
            for (i = 0; i < num_elements; ++i)
            {
                ((float*)(mem->data))[i] *= (float)value;
            }
        }
        else if (location == OSKAR_LOCATION_GPU)
        {
            num_threads = 256;
            num_blocks  = (num_elements + num_threads - 1) / num_threads;
            oskar_cudak_scale_real_f OSKAR_CUDAK_CONF(num_blocks, num_threads)
                    (num_elements, value, (float*)(mem->data));
            cudaDeviceSynchronize();
            error = cudaPeekAtLastError();
        }
        else
        {
            return OSKAR_ERR_BAD_LOCATION;
        }
    }
    else if (oskar_mem_is_double(type))
    {
        if (location == OSKAR_LOCATION_CPU)
        {
            for (i = 0; i < num_elements; ++i)
            {
                ((double*)(mem->data))[i] *= value;
            }
        }
        else if (location == OSKAR_LOCATION_GPU)
        {
            num_threads = 256;
            num_blocks  = (num_elements + num_threads - 1) / num_threads;
            oskar_cudak_scale_real_d OSKAR_CUDAK_CONF(num_blocks, num_threads)
                    (num_elements, value, (double*)(mem->data));
            cudaDeviceSynchronize();
            error = cudaPeekAtLastError();
        }
        else
        {
            return OSKAR_ERR_BAD_LOCATION;
        }
    }
    else
    {
        return OSKAR_ERR_BAD_DATA_TYPE;
    }

    return error;
}
