/*
 * Copyright (c) 2016, The University of Oxford
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

#include "oskar_global.h"
#include "imager/private_imager_generate_w_phase_screen_cuda.h"
#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_imager_generate_w_phase_screen_cudak_f(const int conv_size,
        const int inner_half, const float sampling, const float f,
        const float* restrict taper_func, float* restrict scr)
{
    int ind, offset;
    float l, m, phase, rsq, taper, sin_phase, cos_phase;
    const int ix = blockDim.x * blockIdx.x + threadIdx.x - inner_half;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y - inner_half;
    if (ix >= inner_half || iy >= inner_half) return;

    taper = taper_func[ix + inner_half] * taper_func[iy + inner_half];
    l = sampling * (float)ix;
    m = sampling * (float)iy;
    rsq = l*l + m*m;
    phase = f * (sqrtf(1.0f - rsq) - 1.0f);
    sincosf(phase, &sin_phase, &cos_phase);
    offset = (iy >= 0 ? iy : (iy + conv_size)) * conv_size;
    ind = 2 * (offset + (ix >= 0 ? ix : (ix + conv_size)));
    if (rsq < 1.0f)
    {
        scr[ind]     = taper * cos_phase;
        scr[ind + 1] = taper * sin_phase;
    }
}


/* Double precision. */
__global__
void oskar_imager_generate_w_phase_screen_cudak_d(const int conv_size,
        const int inner_half, const double sampling, const double f,
        const double* restrict taper_func, double* restrict scr)
{
    int ind, offset;
    double l, m, phase, rsq, taper, sin_phase, cos_phase;
    const int ix = blockDim.x * blockIdx.x + threadIdx.x - inner_half;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y - inner_half;
    if (ix >= inner_half || iy >= inner_half) return;

    taper = taper_func[ix + inner_half] * taper_func[iy + inner_half];
    l = sampling * (double)ix;
    m = sampling * (double)iy;
    rsq = l*l + m*m;
    phase = f * (sqrt(1.0 - rsq) - 1.0);
    sincos(phase, &sin_phase, &cos_phase);
    offset = (iy >= 0 ? iy : (iy + conv_size)) * conv_size;
    ind = 2 * (offset + (ix >= 0 ? ix : (ix + conv_size)));
    if (rsq < 1.0)
    {
        scr[ind]     = taper * cos_phase;
        scr[ind + 1] = taper * sin_phase;
    }
}

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_imager_generate_w_phase_screen_cuda_f(int iw, int conv_size,
        int inner, float sampling, float w_scale, const float* taper_func,
        float* scr)
{
    const double f = (2.0 * M_PI * iw * iw) / w_scale;
    const int inner_half = inner / 2;
    const dim3 num_threads(16, 16);
    const dim3 num_blocks((inner + num_threads.x - 1) / num_threads.x,
            (inner + num_threads.y - 1) / num_threads.y);

    oskar_imager_generate_w_phase_screen_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (conv_size, inner_half,
            sampling, (float) f, taper_func, scr);
}

/* Double precision. */
void oskar_imager_generate_w_phase_screen_cuda_d(int iw, int conv_size,
        int inner, double sampling, double w_scale, const double* taper_func,
        double* scr)
{
    const double f = (2.0 * M_PI * iw * iw) / w_scale;
    const int inner_half = inner / 2;
    const dim3 num_threads(16, 16);
    const dim3 num_blocks((inner + num_threads.x - 1) / num_threads.x,
            (inner + num_threads.y - 1) / num_threads.y);

    oskar_imager_generate_w_phase_screen_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (conv_size, inner_half,
            sampling, f, taper_func, scr);
}

#ifdef __cplusplus
}
#endif
