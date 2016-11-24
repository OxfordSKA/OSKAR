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

#include "imager/oskar_imager.h"
#include "imager/oskar_imager_select_vis.h"

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
        const oskar_Mem* weight_in, int im_time_idx, int im_chan_idx,
        int im_pol, oskar_Mem* data_out, oskar_Mem* weight_out,
        size_t* num, int* status)
{
    int t, c, nb, num_chan, pol_offset;
    num_chan = 1 + end_chan - start_chan;
    nb = num_baselines;
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
        t = im_time_idx + h->vis_time_range[0];
        if (t < start_time || t > end_time) return;
        c = im_chan_idx + h->vis_chan_range[0];
        if (c < start_chan || c > end_chan) return;
        copy_vis_pol(data_out, weight_out, *num, data, weight_in,
                ((t - start_time) * num_chan + (c - start_chan)) * nb,
                (t - start_time) * nb,
                nb, num_pols, pol_offset, status);
        *num += nb;
    }
    else if (h->time_snaps && !h->chan_snaps)
    {
        /* Frequency synthesis. */
        t = im_time_idx + h->vis_time_range[0];
        if (t < start_time || t > end_time) return;
        for (c = h->vis_chan_range[0]; c <= h->vis_chan_range[1]; ++c)
        {
            if (c < start_chan || c > end_chan) continue;
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
        c = im_chan_idx + h->vis_chan_range[0];
        if (c < start_chan || c > end_chan) return;
        for (t = h->vis_time_range[0]; t <= h->vis_time_range[1]; ++t)
        {
            if (t < start_time || t > end_time) continue;
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
        for (t = h->vis_time_range[0]; t <= h->vis_time_range[1]; ++t)
        {
            if (t < start_time || t > end_time) continue;
            for (c = h->vis_chan_range[0]; c <= h->vis_chan_range[1]; ++c)
            {
                if (c < start_chan || c > end_chan) continue;
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
