/*
 * Copyright (c) 2019, The University of Oxford
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
#include <cufft.h>
#endif

#include "math/oskar_fft.h"
#include "math/oskar_fftpack_cfft.h"
#include "math/oskar_fftpack_cfft_f.h"

#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_FFT
{
    size_t num_cells_total;
    oskar_Mem *fftpack_work, *fftpack_wsave;
    int precision, location, num_dim, dim_size, ensure_consistent_norm;
#ifdef OSKAR_HAVE_CUDA
    cufftHandle cufft_plan;
#endif
};

oskar_FFT* oskar_fft_create(int precision, int location, int num_dim,
        int dim_size, int batch_size_1d, int* status)
{
    int i;
    oskar_FFT* h = (oskar_FFT*) calloc(1, sizeof(oskar_FFT));
#ifndef OSKAR_HAVE_CUDA
    if (location == OSKAR_GPU) location = OSKAR_CPU;
#endif
#ifndef OSKAR_HAVE_OPENCL
    if (location & OSKAR_CL) location = OSKAR_CPU;
#endif
    h->precision = precision;
    h->location = location;
    h->num_dim = num_dim;
    h->dim_size = dim_size;
    h->ensure_consistent_norm = 1;
    h->num_cells_total = (size_t) dim_size;
    for (i = 1; i < num_dim; ++i) h->num_cells_total *= (size_t) dim_size;
    if (location == OSKAR_CPU)
    {
        int len = 4 * dim_size +
                2 * (int)(log((double)dim_size) / log(2.0)) + 8;
        h->fftpack_wsave = oskar_mem_create(precision, location, len, status);
        if (num_dim == 1)
        {
            (void) batch_size_1d;
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
        }
        else if (num_dim == 2)
        {
            if (precision == OSKAR_DOUBLE)
                oskar_fftpack_cfft2i(dim_size, dim_size,
                        oskar_mem_double(h->fftpack_wsave, status));
            else
                oskar_fftpack_cfft2i_f(dim_size, dim_size,
                        oskar_mem_float(h->fftpack_wsave, status));
        }
        else
            *status = OSKAR_ERR_INVALID_ARGUMENT;
        h->fftpack_work = oskar_mem_create(precision, location,
                2 * h->num_cells_total, status);
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (num_dim == 1)
            cufftPlan1d(&h->cufft_plan, dim_size,
                    ((precision == OSKAR_DOUBLE) ? CUFFT_Z2Z : CUFFT_C2C),
                    batch_size_1d);
        else if (num_dim == 2)
            cufftPlan2d(&h->cufft_plan, dim_size, dim_size,
                    ((precision == OSKAR_DOUBLE) ? CUFFT_Z2Z : CUFFT_C2C));
        else
            *status = OSKAR_ERR_INVALID_ARGUMENT;
#endif
    }
    else if (location & OSKAR_CL)
    {
#ifdef OSKAR_HAVE_OPENCL
        *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
#endif
    }
    else
        *status = OSKAR_ERR_BAD_LOCATION;
    return h;
}

void oskar_fft_exec(oskar_FFT* h, oskar_Mem* data, int* status)
{
    oskar_Mem *data_copy = 0, *data_ptr = data;
    if (*status) return;
    if (oskar_mem_location(data) != h->location)
    {
        data_copy = oskar_mem_create_copy(data, h->location, status);
        data_ptr = data_copy;
    }
    if (h->location == OSKAR_CPU)
    {
        if (h->num_dim == 1)
        {
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
        }
        else if (h->num_dim == 2)
        {
            if (h->precision == OSKAR_DOUBLE)
                oskar_fftpack_cfft2f(h->dim_size, h->dim_size, h->dim_size,
                        oskar_mem_double(data_ptr, status),
                        oskar_mem_double(h->fftpack_wsave, status),
                        oskar_mem_double(h->fftpack_work, status));
            else
                oskar_fftpack_cfft2f_f(h->dim_size, h->dim_size, h->dim_size,
                        oskar_mem_float(data_ptr, status),
                        oskar_mem_float(h->fftpack_wsave, status),
                        oskar_mem_float(h->fftpack_work, status));
            /* This step not needed for W-kernel generation, so turn it off. */
            if (h->ensure_consistent_norm)
                oskar_mem_scale_real(data_ptr, (double)h->num_cells_total,
                        0, h->num_cells_total, status);
        }
    }
    else if (h->location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (h->precision == OSKAR_DOUBLE)
            cufftExecZ2Z(h->cufft_plan,
                    (cufftDoubleComplex*) oskar_mem_void(data_ptr),
                    (cufftDoubleComplex*) oskar_mem_void(data_ptr),
                    CUFFT_FORWARD);
        else
            cufftExecC2C(h->cufft_plan,
                    (cufftComplex*) oskar_mem_void(data_ptr),
                    (cufftComplex*) oskar_mem_void(data_ptr),
                    CUFFT_FORWARD);
#endif
    }
    else
        *status = OSKAR_ERR_BAD_LOCATION;
    if (oskar_mem_location(data) != h->location)
        oskar_mem_copy(data, data_ptr, status);
    oskar_mem_free(data_copy, status);
}

void oskar_fft_free(oskar_FFT* h)
{
    int status = 0;
    if (!h) return;
    oskar_mem_free(h->fftpack_work, &status);
    oskar_mem_free(h->fftpack_wsave, &status);
#ifdef OSKAR_HAVE_CUDA
    if (h->location == OSKAR_GPU)
        cufftDestroy(h->cufft_plan);
#endif
    free(h);
}

void oskar_fft_set_ensure_consistent_norm(oskar_FFT* h, int value)
{
    h->ensure_consistent_norm = value;
}

#ifdef __cplusplus
}
#endif
