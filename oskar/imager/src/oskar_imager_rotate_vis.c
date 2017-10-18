/*
 * Copyright (c) 2016-2017, The University of Oxford
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

#include "math/oskar_cmath.h"
#include "imager/private_imager.h"
#include "imager/oskar_imager.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_rotate_vis(const oskar_Imager* h, size_t num_vis,
        const oskar_Mem* uu_in, const oskar_Mem* vv_in, const oskar_Mem* ww_in,
        oskar_Mem* amps)
{
#ifdef OSKAR_OS_WIN
    int i;
    const int num = (const int) num_vis;
#else
    size_t i;
    const size_t num = num_vis;
#endif
    const double delta_l = h->delta_l;
    const double delta_m = h->delta_m;
    const double delta_n = h->delta_n;
    const double twopi = 2.0 * M_PI;

    if (oskar_mem_precision(amps) == OSKAR_DOUBLE)
    {
        const double *u, *v, *w;
        double2* a;
        u = (const double*)oskar_mem_void_const(uu_in);
        v = (const double*)oskar_mem_void_const(vv_in);
        w = (const double*)oskar_mem_void_const(ww_in);
        a = (double2*)oskar_mem_void(amps);

#pragma omp parallel for private(i)
        for (i = 0; i < num; ++i)
        {
            double arg, phase_re, phase_im, re, im;
            arg = twopi * (u[i] * delta_l + v[i] * delta_m + w[i] * delta_n);
            phase_re = cos(arg);
            phase_im = sin(arg);
            re = a[i].x * phase_re - a[i].y * phase_im;
            im = a[i].x * phase_im + a[i].y * phase_re;
            a[i].x = re;
            a[i].y = im;
        }
    }
    else
    {
        const float *u, *v, *w;
        float2* a;
        u = (const float*)oskar_mem_void_const(uu_in);
        v = (const float*)oskar_mem_void_const(vv_in);
        w = (const float*)oskar_mem_void_const(ww_in);
        a = (float2*)oskar_mem_void(amps);

#pragma omp parallel for private(i)
        for (i = 0; i < num; ++i)
        {
            double arg, phase_re, phase_im, re, im;
            arg = twopi * (u[i] * delta_l + v[i] * delta_m + w[i] * delta_n);
            phase_re = cos(arg);
            phase_im = sin(arg);
            re = a[i].x * phase_re - a[i].y * phase_im;
            im = a[i].x * phase_im + a[i].y * phase_re;
            a[i].x = (float) re;
            a[i].y = (float) im;
        }
    }
}

#ifdef __cplusplus
}
#endif
