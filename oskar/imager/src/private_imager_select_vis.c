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

#include "imager/private_imager.h"
#include "imager/oskar_imager.h"
#include "imager/private_imager_select_vis.h"
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

static void copy_vis_pol(oskar_Mem* amps, oskar_Mem* wt, int amps_offset,
        const oskar_Mem* vis, const oskar_Mem* weight, int vis_offset,
        int weight_offset, int num_baselines, int stride, int pol_offset,
        int* status);

/* Get a set of visibility amplitudes for a single polarisation. */
void oskar_imager_select_vis(const oskar_Imager* h,
        int start_time, int end_time, int start_chan, int end_chan,
        int num_baselines, int num_pols, const oskar_Mem* data,
        const oskar_Mem* weight_in, double im_time_utc, double im_freq_hz,
        int im_pol, oskar_Mem* data_out, oskar_Mem* weight_out,
        size_t* num, int* status)
{
    int i, j, t, c, pol_offset;
    const int num_chan = 1 + end_chan - start_chan;
    const int nb = num_baselines;
    const double s = 0.05;
    const double df = h->freq_inc_hz != 0.0 ? h->freq_inc_hz : 1.0;
    const double dt = h->time_inc_sec != 0.0 ? h->time_inc_sec / 86400.0 : 1.0;
    const double f0 = h->vis_freq_start_hz;
    const double t0 = h->vis_time_start_mjd_utc + 0.5 * dt;
    *num = 0;

    /* Override pol_offset if required. */
    pol_offset = h->pol_offset;
    if (h->im_type == OSKAR_IMAGE_TYPE_STOKES ||
            h->im_type == OSKAR_IMAGE_TYPE_LINEAR)
        pol_offset = im_pol;
    if (num_pols == 1) pol_offset = 0;

    if (h->time_snaps && h->chan_snaps)
    {
        /* Snapshots. */
        t = (int) round((im_time_utc - t0) / dt);
        if (t < start_time || t > end_time) return;
        if (fabs((im_time_utc - t0) - t * dt) > s * dt) return;
        c = (int) round((im_freq_hz - f0) / df);
        if (c < start_chan || c > end_chan) return;
        if (fabs((im_freq_hz - f0) - c * df) > s * df) return;
        copy_vis_pol(data_out, weight_out, *num, data, weight_in,
                ((t - start_time) * num_chan + (c - start_chan)) * nb,
                (t - start_time) * nb,
                nb, num_pols, pol_offset, status);
        *num += nb;
    }
    else if (h->time_snaps && !h->chan_snaps)
    {
        /* Frequency synthesis. */
        t = (int) round((im_time_utc - t0) / dt);
        if (t < start_time || t > end_time) return;
        if (fabs((im_time_utc - t0) - t * dt) > s * dt) return;
        for (i = 0; i < h->num_sel_freqs; ++i)
        {
            c = (int) round((h->sel_freqs[i] - f0) / df);
            if (c < start_chan || c > end_chan) continue;
            if (fabs((h->sel_freqs[i] - f0) - c * df) > s * df) continue;
            copy_vis_pol(data_out, weight_out, *num, data, weight_in,
                    ((t - start_time) * num_chan + (c - start_chan)) * nb,
                    (t - start_time) * nb,
                    nb, num_pols, pol_offset, status);
            *num += nb;
        }
    }
    else if (!h->time_snaps && h->chan_snaps)
    {
        /* Time synthesis. */
        c = (int) round((im_freq_hz - f0) / df);
        if (c < start_chan || c > end_chan) return;
        if (fabs((im_freq_hz - f0) - c * df) > s * df) return;
        for (i = 0; i < h->num_sel_times; ++i)
        {
            t = (int) round((h->sel_times[i] - t0) / dt);
            if (t < start_time || t > end_time) continue;
            if (fabs((h->sel_times[i] - t0) - t * dt) > s * dt) continue;
            copy_vis_pol(data_out, weight_out, *num, data, weight_in,
                    ((t - start_time) * num_chan + (c - start_chan)) * nb,
                    (t - start_time) * nb,
                    nb, num_pols, pol_offset, status);
            *num += nb;
        }
    }
    else
    {
        /* Time and frequency synthesis. */
        for (i = 0; i < h->num_sel_times; ++i)
        {
            t = (int) round((h->sel_times[i] - t0) / dt);
            if (t < start_time || t > end_time) continue;
            if (fabs((h->sel_times[i] - t0) - t * dt) > s * dt) continue;
            for (j = 0; j < h->num_sel_freqs; ++j)
            {
                c = (int) round((h->sel_freqs[j] - f0) / df);
                if (c < start_chan || c > end_chan) continue;
                if (fabs((h->sel_freqs[j] - f0) - c * df) > s * df) continue;
                copy_vis_pol(data_out, weight_out, *num, data, weight_in,
                        ((t - start_time) * num_chan + (c - start_chan)) * nb,
                        (t - start_time) * nb,
                        nb, num_pols, pol_offset, status);
                *num += nb;
            }
        }
    }
}

void copy_vis_pol(oskar_Mem* amps, oskar_Mem* wt, int amps_offset,
        const oskar_Mem* vis, const oskar_Mem* weight, int vis_offset,
        int weight_offset, int num_baselines, int stride, int pol_offset,
        int* status)
{
    int i;
    if (*status) return;
    if (oskar_mem_precision(amps) == OSKAR_SINGLE)
    {
        float* w_out;
        const float* w_in;
        w_out = oskar_mem_float(wt, status) + amps_offset;
        w_in = oskar_mem_float_const(weight, status);
        for (i = 0; i < num_baselines; ++i)
            w_out[i] = w_in[stride * (weight_offset + i) + pol_offset];

        if (vis)
        {
            float2* a;
            const float2* v;
            a = oskar_mem_float2(amps, status) + amps_offset;
            v = oskar_mem_float2_const(vis, status);
            for (i = 0; i < num_baselines; ++i)
                a[i] = v[stride * (vis_offset + i) + pol_offset];
        }
    }
    else
    {
        double* w_out;
        const double* w_in;
        w_out = oskar_mem_double(wt, status) + amps_offset;
        w_in = oskar_mem_double_const(weight, status);
        for (i = 0; i < num_baselines; ++i)
            w_out[i] = w_in[stride * (weight_offset + i) + pol_offset];

        if (vis)
        {
            double2* a;
            const double2* v;
            a = oskar_mem_double2(amps, status) + amps_offset;
            v = oskar_mem_double2_const(vis, status);
            for (i = 0; i < num_baselines; ++i)
                a[i] = v[stride * (vis_offset + i) + pol_offset];
        }
    }
}


#ifdef __cplusplus
}
#endif
