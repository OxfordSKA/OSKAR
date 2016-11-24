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

#include "math/oskar_cmath.h"
#include "imager/oskar_imager_rotate_vis.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_rotate_vis(size_t num_vis, const oskar_Mem* uu,
        const oskar_Mem* vv, const oskar_Mem* ww, oskar_Mem* amps,
        const double delta_l, const double delta_m, const double delta_n)
{
    size_t i;

    if (oskar_mem_precision(amps) == OSKAR_DOUBLE)
    {
        const double *u, *v, *w;
        double2* a;
        u = (const double*)oskar_mem_void_const(uu);
        v = (const double*)oskar_mem_void_const(vv);
        w = (const double*)oskar_mem_void_const(ww);
        a = (double2*)oskar_mem_void(amps);

        for (i = 0; i < num_vis; ++i)
        {
            double arg, phase_re, phase_im, re, im;
            arg = 2.0*M_PI * (u[i] * delta_l + v[i] * delta_m + w[i] * delta_n);
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
        u = (const float*)oskar_mem_void_const(uu);
        v = (const float*)oskar_mem_void_const(vv);
        w = (const float*)oskar_mem_void_const(ww);
        a = (float2*)oskar_mem_void(amps);

        for (i = 0; i < num_vis; ++i)
        {
            float arg, phase_re, phase_im, re, im;
            arg = 2.0*M_PI * (u[i] * delta_l + v[i] * delta_m + w[i] * delta_n);
            phase_re = cos(arg);
            phase_im = sin(arg);
            re = a[i].x * phase_re - a[i].y * phase_im;
            im = a[i].x * phase_im + a[i].y * phase_re;
            a[i].x = re;
            a[i].y = im;
        }
    }
}

#ifdef __cplusplus
}
#endif
