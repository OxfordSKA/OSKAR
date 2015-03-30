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

#include <private_mem.h>
#include <oskar_mem.h>
#include <oskar_cuda_check_error.h>
#include <oskar_mem_scale_real_cuda.h>

#ifdef __cplusplus
extern "C"
#endif
void oskar_mem_scale_real(oskar_Mem* mem, double value, int* status)
{
    size_t num_elements, i;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get memory meta-data. */
    num_elements = mem->num_elements;

    /* Check if elements are real, complex or matrix. */
    if (oskar_type_is_complex(mem->type))
        num_elements *= 2;
    if (oskar_type_is_matrix(mem->type))
        num_elements *= 4;

    /* Scale the vector. */
    if (oskar_type_is_single(mem->type))
    {
        if (mem->location == OSKAR_CPU)
        {
            float *aa;
            aa = (float*) mem->data;
            for (i = 0; i < num_elements; ++i) aa[i] *= (float)value;
        }
        else if (mem->location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_mem_scale_real_cuda_f(num_elements, (float)value,
                    (float*)(mem->data));
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
        {
            *status = OSKAR_ERR_BAD_LOCATION;
        }
    }
    else if (oskar_type_is_double(mem->type))
    {
        if (mem->location == OSKAR_CPU)
        {
            double *aa;
            aa = (double*) mem->data;
            for (i = 0; i < num_elements; ++i) aa[i] *= value;
        }
        else if (mem->location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_mem_scale_real_cuda_d(num_elements, value,
                    (double*)(mem->data));
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
        {
            *status = OSKAR_ERR_BAD_LOCATION;
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
}
