/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <private_mem.h>
#include <oskar_mem.h>

#include <oskar_mem_set_value_real_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_set_value_real(oskar_Mem* mem, double val,
        size_t offset, size_t length, int* status)
{
    size_t i, n;
    int type, location;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the data type, location, and number of elements. */
    type = mem->type;
    location = mem->location;
    n = length;
    if (offset == 0 && length == 0)
    {
        n = mem->num_elements;
    }

    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_DOUBLE)
        {
            double *v;
            v = (double*)(mem->data) + offset;
            for (i = 0; i < n; ++i) v[i] = val;
        }
        else if (type == OSKAR_DOUBLE_COMPLEX)
        {
            double2 *v;
            v = (double2*)(mem->data) + offset;
            for (i = 0; i < n; ++i)
            {
                v[i].x = val;
                v[i].y = 0.0;
            }
        }
        else if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
        {
            double4c d;
            double4c *v;
            v = (double4c*)(mem->data) + offset;
            for (i = 0; i < n; ++i)
            {
                d.a.x = val;
                d.a.y = 0.0;
                d.b.x = 0.0;
                d.b.y = 0.0;
                d.c.x = 0.0;
                d.c.y = 0.0;
                d.d.x = val;
                d.d.y = 0.0;
                v[i] = d;
            }
        }
        else if (type == OSKAR_SINGLE)
        {
            float *v;
            v = (float*)(mem->data) + offset;
            for (i = 0; i < n; ++i) v[i] = (float)val;
        }
        else if (type == OSKAR_SINGLE_COMPLEX)
        {
            float2 *v;
            v = (float2*)(mem->data) + offset;
            for (i = 0; i < n; ++i)
            {
                v[i].x = (float)val;
                v[i].y = 0.0f;
            }
        }
        else if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
        {
            float4c d;
            float4c *v;
            v = (float4c*)(mem->data) + offset;
            for (i = 0; i < n; ++i)
            {
                d.a.x = (float)val;
                d.a.y = 0.0f;
                d.b.x = 0.0f;
                d.b.y = 0.0f;
                d.c.x = 0.0f;
                d.c.y = 0.0f;
                d.d.x = (float)val;
                d.d.y = 0.0f;
                v[i] = d;
            }
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_DOUBLE)
        {
            oskar_mem_set_value_real_cuda_r_d(n,
                    (double*)(mem->data) + offset, val);
        }
        else if (type == OSKAR_DOUBLE_COMPLEX)
        {
            oskar_mem_set_value_real_cuda_c_d(n,
                    (double2*)(mem->data) + offset, val);
        }
        else if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
        {
            oskar_mem_set_value_real_cuda_m_d(n,
                    (double4c*)(mem->data) + offset, val);
        }
        else if (type == OSKAR_SINGLE)
        {
            oskar_mem_set_value_real_cuda_r_f(n,
                    (float*)(mem->data) + offset, (float)val);
        }
        else if (type == OSKAR_SINGLE_COMPLEX)
        {
            oskar_mem_set_value_real_cuda_c_f(n,
                    (float2*)(mem->data) + offset, (float)val);
        }
        else if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
        {
            oskar_mem_set_value_real_cuda_m_f(n,
                    (float4c*)(mem->data) + offset, (float)val);
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
