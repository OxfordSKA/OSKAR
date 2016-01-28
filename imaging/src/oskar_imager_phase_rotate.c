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

#include <oskar_cmath.h>
#include <oskar_imager_phase_rotate.h>

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_phase_rotate(size_t num_vis, const oskar_Mem* uu,
        const oskar_Mem* vv, const oskar_Mem* ww, oskar_Mem* amps,
        double delta_l, double delta_m, double delta_n)
{
    size_t i;
    const double scale_wavenumber = 2.0 * M_PI;

    if (oskar_mem_precision(amps) == OSKAR_DOUBLE)
    {
        const double *uu_, *vv_, *ww_;
        double2* amp_;
        uu_ = (const double*)oskar_mem_void_const(uu);
        vv_ = (const double*)oskar_mem_void_const(vv);
        ww_ = (const double*)oskar_mem_void_const(ww);
        amp_ = (double2*)oskar_mem_void(amps);

        for (i = 0; i < num_vis; ++i)
        {
            double u, v, w, arg, phase_re, phase_im, re, im;
            u = uu_[i] * scale_wavenumber;
            v = vv_[i] * scale_wavenumber;
            w = ww_[i] * scale_wavenumber;
            arg = u * delta_l + v * delta_m + w * delta_n;
            phase_re = cos(arg);
            phase_im = sin(arg);
            re = amp_[i].x * phase_re - amp_[i].y * phase_im;
            im = amp_[i].x * phase_im + amp_[i].y * phase_re;
            amp_[i].x = re;
            amp_[i].y = im;
        }
    }
    else
    {
        const float *uu_, *vv_, *ww_;
        float2* amp_;
        uu_ = (const float*)oskar_mem_void_const(uu);
        vv_ = (const float*)oskar_mem_void_const(vv);
        ww_ = (const float*)oskar_mem_void_const(ww);
        amp_ = (float2*)oskar_mem_void(amps);

        for (i = 0; i < num_vis; ++i)
        {
            float u, v, w, arg, phase_re, phase_im, re, im;
            u = uu_[i] * scale_wavenumber;
            v = vv_[i] * scale_wavenumber;
            w = ww_[i] * scale_wavenumber;
            arg = u * delta_l + v * delta_m + w * delta_n;
            phase_re = cos(arg);
            phase_im = sin(arg);
            re = amp_[i].x * phase_re - amp_[i].y * phase_im;
            im = amp_[i].x * phase_im + amp_[i].y * phase_re;
            amp_[i].x = re;
            amp_[i].y = im;
        }
    }
}

#ifdef __cplusplus
}
#endif
