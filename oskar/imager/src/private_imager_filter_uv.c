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

#include "imager/private_imager.h"

#include "imager/private_imager_filter_uv.h"
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_filter_uv(const oskar_Imager* h, size_t* num_vis,
        oskar_Mem* uu, oskar_Mem* vv, oskar_Mem* ww, oskar_Mem* amp,
        oskar_Mem* weight, int* status)
{
    size_t i, n;
    double r, range[2];

    /* Return immediately if filtering is not enabled. */
    if (h->uv_filter_min <= 0.0 && h->uv_filter_max < 0.0) return;
    if (*status) return;

    /* Get the range (squared, to avoid lots of square roots later). */
    range[0] = h->uv_filter_min;
    range[1] = (h->uv_filter_max < 0.0) ? (double) FLT_MAX : h->uv_filter_max;
    range[0] *= range[0];
    range[1] *= range[1];

    /* Get the number of input points, and set the number selected to zero. */
    n = *num_vis;
    *num_vis = 0;

    /* Apply the UV baseline length filter. */
    if (h->imager_prec == OSKAR_DOUBLE)
    {
        double *uu_, *vv_, *ww_, *weight_;
        double2* amp_;
        uu_ = oskar_mem_double(uu, status);
        vv_ = oskar_mem_double(vv, status);
        ww_ = oskar_mem_double(ww, status);
        amp_ = oskar_mem_double2(amp, status);
        weight_ = oskar_mem_double(weight, status);

        for (i = 0; i < n; ++i)
        {
            r = uu_[i] * uu_[i] + vv_[i] * vv_[i];
            if (r >= range[0] && r <= range[1])
            {
                uu_[*num_vis] = uu_[i];
                vv_[*num_vis] = vv_[i];
                ww_[*num_vis] = ww_[i];
                amp_[*num_vis] = amp_[i];
                weight_[*num_vis] = weight_[i];
                (*num_vis)++;
            }
        }
    }
    else
    {
        float *uu_, *vv_, *ww_, *weight_;
        float2* amp_;
        uu_ = oskar_mem_float(uu, status);
        vv_ = oskar_mem_float(vv, status);
        ww_ = oskar_mem_float(ww, status);
        amp_ = oskar_mem_float2(amp, status);
        weight_ = oskar_mem_float(weight, status);

        for (i = 0; i < n; ++i)
        {
            r = uu_[i] * uu_[i] + vv_[i] * vv_[i];
            if (r >= range[0] && r <= range[1])
            {
                uu_[*num_vis] = uu_[i];
                vv_[*num_vis] = vv_[i];
                ww_[*num_vis] = ww_[i];
                amp_[*num_vis] = amp_[i];
                weight_[*num_vis] = weight_[i];
                (*num_vis)++;
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
