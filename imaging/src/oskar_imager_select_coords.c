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

#include <private_imager.h>

#include <oskar_imager_select_coords.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_select_coords(const oskar_Imager* h,
        int start_time, int end_time, int start_chan, int end_chan,
        int num_baselines, const oskar_Mem* vis_uu, const oskar_Mem* vis_vv,
        const oskar_Mem* vis_ww, int im_time_idx, int im_chan_idx,
        oskar_Mem* uu, oskar_Mem* vv, oskar_Mem* ww, size_t* num, int* status)
{
    int c, t;
    double freq, scaling, c0 = 299792458.0;
    oskar_Mem *uu_, *vv_, *ww_;
    size_t src;
    if (*status) return;
    *num = 0;

    /* Check whether using time and/or frequency synthesis. */
    if (h->time_snaps && h->chan_snaps)
    {
        /* Get the time and channel for the image and check if out of range. */
        t = im_time_idx + h->vis_time_range[0];
        if (t < start_time || t > end_time) return;
        c = im_chan_idx + h->vis_chan_range[0];
        if (c < start_chan || c > end_chan) return;

        /* Copy the baseline coordinates for the selected time. */
        src = num_baselines * (t - start_time);
        oskar_mem_copy_contents(uu, vis_uu, 0, src, num_baselines, status);
        oskar_mem_copy_contents(vv, vis_vv, 0, src, num_baselines, status);
        oskar_mem_copy_contents(ww, vis_ww, 0, src, num_baselines, status);

        /* Divide coordinates by the wavelength. */
        freq = h->vis_freq_start_hz + c * h->freq_inc_hz;
        scaling = freq / c0;
        oskar_mem_scale_real(uu, scaling, status);
        oskar_mem_scale_real(vv, scaling, status);
        oskar_mem_scale_real(ww, scaling, status);
        *num += num_baselines;
    }
    else if (h->time_snaps && !h->chan_snaps) /* Frequency synthesis */
    {
        /* Get the time for the image and check if out of range. */
        t = im_time_idx + h->vis_time_range[0];
        if (t < start_time || t > end_time) return;

        /* Copy the baseline coordinates for the selected time. */
        uu_ = oskar_mem_create_alias(0, 0, 0, status);
        vv_ = oskar_mem_create_alias(0, 0, 0, status);
        ww_ = oskar_mem_create_alias(0, 0, 0, status);
        for (c = h->vis_chan_range[0]; c <= h->vis_chan_range[1]; ++c)
        {
            if (c < start_chan || c > end_chan) continue;

            src = num_baselines * (t - start_time);
            oskar_mem_set_alias(uu_, uu, *num, num_baselines, status);
            oskar_mem_set_alias(vv_, vv, *num, num_baselines, status);
            oskar_mem_set_alias(ww_, ww, *num, num_baselines, status);
            oskar_mem_copy_contents(uu_, vis_uu, 0, src,
                    num_baselines, status);
            oskar_mem_copy_contents(vv_, vis_vv, 0, src,
                    num_baselines, status);
            oskar_mem_copy_contents(ww_, vis_ww, 0, src,
                    num_baselines, status);

            /* Divide coordinates by the wavelength. */
            freq = h->vis_freq_start_hz + c * h->freq_inc_hz;
            scaling = freq / c0;
            oskar_mem_scale_real(uu_, scaling, status);
            oskar_mem_scale_real(vv_, scaling, status);
            oskar_mem_scale_real(ww_, scaling, status);
            *num += num_baselines;
        }
        oskar_mem_free(uu_, status);
        oskar_mem_free(vv_, status);
        oskar_mem_free(ww_, status);
    }
    else if (!h->time_snaps && h->chan_snaps) /* Time synthesis */
    {
        c = im_chan_idx + h->vis_chan_range[0];
        if (c < start_chan || c > end_chan) return;
        for (t = h->vis_time_range[0]; t <= h->vis_time_range[1]; ++t)
        {
            if (t < start_time || t > end_time) continue;

            /* Copy the baseline coordinates for the current time. */
            src = num_baselines * (t - start_time);
            oskar_mem_copy_contents(uu, vis_uu, *num, src,
                    num_baselines, status);
            oskar_mem_copy_contents(vv, vis_vv, *num, src,
                    num_baselines, status);
            oskar_mem_copy_contents(ww, vis_ww, *num, src,
                    num_baselines, status);

            /* Divide coordinates by the wavelength. */
            freq = h->vis_freq_start_hz + c * h->freq_inc_hz;
            scaling = freq / c0;
            oskar_mem_scale_real(uu, scaling, status);
            oskar_mem_scale_real(vv, scaling, status);
            oskar_mem_scale_real(ww, scaling, status);
            *num += num_baselines;
        }
    }
    else /* Time and frequency synthesis */
    {
        uu_ = oskar_mem_create_alias(0, 0, 0, status);
        vv_ = oskar_mem_create_alias(0, 0, 0, status);
        ww_ = oskar_mem_create_alias(0, 0, 0, status);
        for (t = h->vis_time_range[0]; t <= h->vis_time_range[1]; ++t)
        {
            if (t < start_time || t > end_time) continue;

            for (c = h->vis_chan_range[0]; c <= h->vis_chan_range[1]; ++c)
            {
                if (c < start_chan || c > end_chan) continue;

                /* Copy the baseline coordinates for the current time. */
                src = num_baselines * (t - start_time);
                oskar_mem_set_alias(uu_, uu, *num, num_baselines, status);
                oskar_mem_set_alias(vv_, vv, *num, num_baselines, status);
                oskar_mem_set_alias(ww_, ww, *num, num_baselines, status);
                oskar_mem_copy_contents(uu_, vis_uu, 0, src,
                        num_baselines, status);
                oskar_mem_copy_contents(vv_, vis_vv, 0, src,
                        num_baselines, status);
                oskar_mem_copy_contents(ww_, vis_ww, 0, src,
                        num_baselines, status);

                /* Divide coordinates by the wavelength. */
                freq = h->vis_freq_start_hz + c * h->freq_inc_hz;
                scaling = freq / c0;
                oskar_mem_scale_real(uu_, scaling, status);
                oskar_mem_scale_real(vv_, scaling, status);
                oskar_mem_scale_real(ww_, scaling, status);
                *num += num_baselines;
            }
        }
        oskar_mem_free(uu_, status);
        oskar_mem_free(vv_, status);
        oskar_mem_free(ww_, status);
    }
}

#ifdef __cplusplus
}
#endif
