/*
 * Copyright (c) 2016-2018, The University of Oxford
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

#include "math/oskar_fftphase.h"
#include "math/oskar_fftphase_cuda.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_fftphase_cf(const int num_x, const int num_y, float* complex_data)
{
    int ix, iy;
    for (iy = 0; iy < num_y; ++iy)
    {
        size_t i1 = iy;
        i1 *= num_x;
        for (ix = 0; ix < num_x; ++ix)
        {
            const size_t i = (i1 + ix) << 1;
            const int x = 1 - (((ix + iy) & 1) << 1);
            complex_data[i]     *= x;
            complex_data[i + 1] *= x;
        }
    }
}

void oskar_fftphase_cd(const int num_x, const int num_y, double* complex_data)
{
    int ix, iy;
    for (iy = 0; iy < num_y; ++iy)
    {
        size_t i1 = iy;
        i1 *= num_x;
        for (ix = 0; ix < num_x; ++ix)
        {
            const size_t i = (i1 + ix) << 1;
            const int x = 1 - (((ix + iy) & 1) << 1);
            complex_data[i]     *= x;
            complex_data[i + 1] *= x;
        }
    }
}

void oskar_fftphase(const int num_x, const int num_y,
        oskar_Mem* complex_data, int* status)
{
    int type, location;
    if (*status) return;
    if (!oskar_mem_is_complex(complex_data))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    type = oskar_mem_precision(complex_data);
    location = oskar_mem_location(complex_data);
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
            oskar_fftphase_cf(num_x, num_y,
                    oskar_mem_float(complex_data, status));
        else if (type == OSKAR_DOUBLE)
            oskar_fftphase_cd(num_x, num_y,
                    oskar_mem_double(complex_data, status));
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_SINGLE)
            oskar_fftphase_cuda_cf(num_x, num_y,
                    oskar_mem_float(complex_data, status));
        else if (type == OSKAR_DOUBLE)
            oskar_fftphase_cuda_cd(num_x, num_y,
                    oskar_mem_double(complex_data, status));
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
