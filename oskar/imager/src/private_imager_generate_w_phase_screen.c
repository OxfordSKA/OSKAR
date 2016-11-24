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

#include "imager/private_imager_generate_w_phase_screen.h"
#include "imager/private_imager_generate_w_phase_screen_cuda.h"
#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif


void oskar_imager_generate_w_phase_screen(const int iw, const int conv_size,
        const int inner, const double sampling, const double w_scale,
        const oskar_Mem* taper_func, oskar_Mem* screen, int* status)
{
    int iy;
    const double f = (2.0 * M_PI * iw * iw) / w_scale;
    const int inner_half = inner / 2;

    oskar_mem_clear_contents(screen, status);
    if (*status) return;
    if (oskar_mem_precision(screen) == OSKAR_SINGLE)
    {
        float* scr;
        const float* tp;
        scr = oskar_mem_float(screen, status);
        tp = oskar_mem_float_const(taper_func, status);
        if (oskar_mem_location(screen) == OSKAR_CPU)
        {
            for (iy = -inner_half; iy < inner_half; ++iy)
            {
                int ix, ind, offset;
                double l, m, msq, phase, rsq, taper, taper_x, taper_y;
                taper_y = tp[iy + inner_half];
                m = sampling * (double)iy;
                msq = m*m;
                offset = (iy > -1 ? iy : (iy + conv_size)) * conv_size;
                for (ix = -inner_half; ix < inner_half; ++ix)
                {
                    l = sampling * (double)ix;
                    rsq = l*l + msq;
                    if (rsq < 1.0)
                    {
                        taper_x = tp[ix + inner_half];
                        taper = taper_x * taper_y;
                        ind = 2 * (offset + (ix > -1 ? ix : (ix + conv_size)));
                        phase = f * (sqrt(1.0 - rsq) - 1.0);
                        scr[ind]     = taper * cos(phase);
                        scr[ind + 1] = taper * sin(phase);
                    }
                }
            }
        }
        else
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_imager_generate_w_phase_screen_cuda_f(iw, conv_size,
                    inner, (float) sampling, (float) w_scale, tp, scr);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
    }
    else
    {
        double* scr;
        const double* tp;
        scr = oskar_mem_double(screen, status);
        tp = oskar_mem_double_const(taper_func, status);
        if (oskar_mem_location(screen) == OSKAR_CPU)
        {
            for (iy = -inner_half; iy < inner_half; ++iy)
            {
                int ix, ind, offset;
                double l, m, msq, phase, rsq, taper, taper_x, taper_y;
                taper_y = tp[iy + inner_half];
                m = sampling * (double)iy;
                msq = m*m;
                offset = (iy > -1 ? iy : (iy + conv_size)) * conv_size;
                for (ix = -inner_half; ix < inner_half; ++ix)
                {
                    l = sampling * (double)ix;
                    rsq = l*l + msq;
                    if (rsq < 1.0)
                    {
                        taper_x = tp[ix + inner_half];
                        taper = taper_x * taper_y;
                        ind = 2 * (offset + (ix > -1 ? ix : (ix + conv_size)));
                        phase = f * (sqrt(1.0 - rsq) - 1.0);
                        scr[ind]     = taper * cos(phase);
                        scr[ind + 1] = taper * sin(phase);
                    }
                }
            }
        }
        else
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_imager_generate_w_phase_screen_cuda_d(iw, conv_size,
                    inner, sampling, w_scale, tp, scr);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
    }
}


#ifdef __cplusplus
}
#endif
