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

#include "imaging/oskar_fft_utility.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_fft_shift_z(const unsigned nx, const unsigned ny, double2* data)
{
    for (unsigned j = 0; j < ny; ++j)
    {
        for (unsigned i = 0; i < nx; ++i)
        {
            if ((i + j) % 2)
            {
                data[j * nx + i].x = -data[j * nx + i].x;
                data[j * nx + i].y = -data[j * nx + i].y;
            }
        }
    }
}


void oskar_fft_shift_d(const unsigned nx, const unsigned ny, double* data)
{
    for (unsigned j = 0; j < ny; ++j)
    {
        for (unsigned i = 0; i < nx; ++i)
        {
            // Note: both equivalent... which is faster
            data[j * nx + i] *= (int) pow(-1, i + j);
//            if ((i + j) % 2)
//            {
//                data[j * nx + i] = -data[j * nx + i];
//            }
        }
    }
}


void oskar_fft_shift_fftz(const unsigned nx, const unsigned ny, fftw_complex* data)
{
    for (unsigned j = 0; j < ny; ++j)
    {
        for (unsigned i = 0; i < nx; ++i)
        {
            // Note: both equivalent... which is faster
            int factor = (int) pow(-1, i + j);
            data[j * nx + i][0] *= factor;
            data[j * nx + i][1] *= factor;
//            if ((i + j) % 2)
//            {
//                data[j * nx + i][0] = -data[j * nx + i][0];
//                data[j * nx + i][1] = -data[j * nx + i][1];
//            }
        }
    }
}


void oskar_fft_z2r_2d(const unsigned size, const double2* in, double* out)
{
    // Convert to half complex form.
    // see:
    // www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data
    unsigned size_hc = (size / 2) + 1;
    size_t mem_size = size * size_hc * sizeof(fftw_complex);
    fftw_complex* fft_in_hc = (fftw_complex*) fftw_malloc(mem_size);
    for (unsigned j = 0; j < size; ++j)
    {
        const void* from = &in[j * size];
        void* to         = &fft_in_hc[j * size_hc];
        memcpy(to, from, size_hc * sizeof(fftw_complex));
    }

    oskar_fft_shift_fftz(size_hc, size, fft_in_hc);
    unsigned int flags = FFTW_ESTIMATE;
    fftw_plan plan = fftw_plan_dft_c2r_2d(size, size, fft_in_hc, out, flags);
    fftw_execute(plan);
    oskar_fft_shift_d(size, size, out);

    fftw_free(fft_in_hc);
    fftw_destroy_plan(plan);
}

#ifdef __cplusplus
}
#endif
