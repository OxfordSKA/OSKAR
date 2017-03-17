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
#include "imager/private_imager_select_coords.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define C0 299792458.0

void oskar_imager_select_coords(const oskar_Imager* h,
        int start_time, int end_time, int start_chan, int end_chan,
        int num_baselines, const oskar_Mem* vis_uu, const oskar_Mem* vis_vv,
        const oskar_Mem* vis_ww, double im_time_utc, double im_freq_hz,
        oskar_Mem* uu, oskar_Mem* vv, oskar_Mem* ww, size_t* num, int* status)
{
    size_t src;
    int i, j, t, c;
    oskar_Mem *uu_ = 0, *vv_ = 0, *ww_ = 0;
    double inv_wavelength;
    const double s = 0.05;
    const double df = h->freq_inc_hz;
    const double dt = h->time_inc_sec / 86400.0;
    const double f0 = h->vis_freq_start_hz;
    const double t0 = h->vis_time_start_mjd_utc + 0.5 * dt;

    /* Initialise. */
    if (*status) return;
    *num = 0;
    uu_ = oskar_mem_create_alias(0, 0, 0, status);
    vv_ = oskar_mem_create_alias(0, 0, 0, status);
    ww_ = oskar_mem_create_alias(0, 0, 0, status);

    /* Check whether using time and/or frequency synthesis. */
    if (h->time_snaps && h->chan_snaps)
    {
        /* Get the time and channel for the image and check if out of range. */
        t = (int) round((im_time_utc - t0) / dt);
        if (t < start_time || t > end_time) return;
        if (fabs((im_time_utc - t0) - t * dt) > s * dt) return;
        c = (int) round((im_freq_hz - f0) / df);
        if (c < start_chan || c > end_chan) return;
        if (fabs((im_freq_hz - f0) - c * df) > s * df) return;

        /* Copy the baseline coordinates for the selected time. */
        src = num_baselines * (t - start_time);
        oskar_mem_copy_contents(uu, vis_uu, 0, src, num_baselines, status);
        oskar_mem_copy_contents(vv, vis_vv, 0, src, num_baselines, status);
        oskar_mem_copy_contents(ww, vis_ww, 0, src, num_baselines, status);

        /* Divide coordinates by the wavelength. */
        inv_wavelength = (f0 + c * df) / C0;
        oskar_mem_scale_real(uu, inv_wavelength, status);
        oskar_mem_scale_real(vv, inv_wavelength, status);
        oskar_mem_scale_real(ww, inv_wavelength, status);
        *num += num_baselines;
    }
    else if (h->time_snaps && !h->chan_snaps) /* Frequency synthesis */
    {
        /* Get the time for the image and check if out of range. */
        t = (int) round((im_time_utc - t0) / dt);
        if (t < start_time || t > end_time) return;
        if (fabs((im_time_utc - t0) - t * dt) > s * dt) return;

        /* Copy the baseline coordinates for the selected time. */
        for (i = 0; i < h->num_sel_freqs; ++i)
        {
            c = (int) round((h->sel_freqs[i] - f0) / df);
            if (c < start_chan || c > end_chan) continue;
            if (fabs((h->sel_freqs[i] - f0) - c * df) > s * df) continue;

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
            inv_wavelength = (f0 + c * df) / C0;
            oskar_mem_scale_real(uu_, inv_wavelength, status);
            oskar_mem_scale_real(vv_, inv_wavelength, status);
            oskar_mem_scale_real(ww_, inv_wavelength, status);
            *num += num_baselines;
        }
    }
    else if (!h->time_snaps && h->chan_snaps) /* Time synthesis */
    {
        /* Get the channel for the image and check if out of range. */
        c = (int) round((im_freq_hz - f0) / df);
        if (c < start_chan || c > end_chan) return;
        if (fabs((im_freq_hz - f0) - c * df) > s * df) return;

        /* Copy the baseline coordinates for all times. */
        for (i = 0; i < h->num_sel_times; ++i)
        {
            t = (int) round((h->sel_times[i] - t0) / dt);
            if (t < start_time || t > end_time) continue;
            if (fabs((h->sel_times[i] - t0) - t * dt) > s * dt) continue;

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
            inv_wavelength = (f0 + c * df) / C0;
            oskar_mem_scale_real(uu_, inv_wavelength, status);
            oskar_mem_scale_real(vv_, inv_wavelength, status);
            oskar_mem_scale_real(ww_, inv_wavelength, status);
            *num += num_baselines;
        }
    }
    else /* Time and frequency synthesis */
    {
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
                inv_wavelength = (f0 + c * df) / C0;
                oskar_mem_scale_real(uu_, inv_wavelength, status);
                oskar_mem_scale_real(vv_, inv_wavelength, status);
                oskar_mem_scale_real(ww_, inv_wavelength, status);
                *num += num_baselines;
            }
        }
    }
    oskar_mem_free(uu_, status);
    oskar_mem_free(vv_, status);
    oskar_mem_free(ww_, status);
}

#ifdef __cplusplus
}
#endif
