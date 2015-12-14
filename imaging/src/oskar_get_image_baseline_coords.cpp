/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#include <oskar_get_image_baseline_coords.h>
#include <oskar_evaluate_image_ranges.h>

#include <oskar_mem.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif


int oskar_get_image_baseline_coords(oskar_Mem* uu, oskar_Mem* vv, oskar_Mem* ww,
        const oskar_Mem* vis_uu, const oskar_Mem* vis_vv, const oskar_Mem* vis_ww,
        int num_times, int num_baselines, int num_channels,
        double freq_start_hz, double freq_inc_hz,int vis_time, double im_freq,
        const oskar_SettingsImage* settings)
{
    int type, err = 0;
    int num_vis_coords = num_baselines;
    int vis_time_range[2], vis_chan_range[2];
    oskar_Mem *uu_ptr, *vv_ptr, *ww_ptr;

    // Data ranges for frequency and time synthesis.
    oskar_evaluate_image_data_range(vis_time_range, settings->time_range,
            num_times, &err);
    oskar_evaluate_image_data_range(vis_chan_range, settings->channel_range,
            num_channels, &err);
    if (err) return err;

    // Declare temporary pointers into visibility coordinate arrays.
    type = oskar_mem_type(vis_uu);
    uu_ptr = oskar_mem_create_alias(vis_uu, 0, num_baselines, &err);
    vv_ptr = oskar_mem_create_alias(vis_vv, 0, num_baselines, &err);
    ww_ptr = oskar_mem_create_alias(vis_ww, 0, num_baselines, &err);
    if (err) return err;

    /* ====================================== TIME SNAPSHOTS, FREQ SNAPSHOTS */
    if (settings->time_snapshots && settings->channel_snapshots)
    {
        size_t byte_offset, data_size;
        data_size = oskar_mem_element_size(type) * num_baselines;
        byte_offset = data_size * vis_time;
        memcpy(oskar_mem_void(uu), (const void*)(oskar_mem_char_const(vis_uu) +
                byte_offset), data_size);
        memcpy(oskar_mem_void(vv), (const void*)(oskar_mem_char_const(vis_vv) +
                byte_offset), data_size);
        memcpy(oskar_mem_void(ww), (const void*)(oskar_mem_char_const(vis_ww) +
                byte_offset), data_size);
    }

    /* ====================================== TIME SNAPSHOTS, FREQ SYNTHESIS */
    else if (settings->time_snapshots && !settings->channel_snapshots)
    {
        int coord_offset = vis_time * num_baselines;
        oskar_mem_set_alias(uu_ptr, vis_uu, coord_offset, num_vis_coords, &err);
        oskar_mem_set_alias(vv_ptr, vis_vv, coord_offset, num_vis_coords, &err);
        oskar_mem_set_alias(ww_ptr, vis_ww, coord_offset, num_vis_coords, &err);
        if (err) return err;

        for (int c = vis_chan_range[0], i = 0; c <= vis_chan_range[1]; ++c)
        {
            double freq = freq_start_hz + c * freq_inc_hz;
            double scaling = freq/im_freq;

            if (type == OSKAR_DOUBLE)
            {
                double *uu_, *vv_, *ww_;
                const double *uu_ptr_, *vv_ptr_, *ww_ptr_;
                uu_ = oskar_mem_double(uu, &err);
                vv_ = oskar_mem_double(vv, &err);
                ww_ = oskar_mem_double(ww, &err);
                uu_ptr_ = oskar_mem_double_const(uu_ptr, &err);
                vv_ptr_ = oskar_mem_double_const(vv_ptr, &err);
                ww_ptr_ = oskar_mem_double_const(ww_ptr, &err);
                for (int b = 0; b < num_baselines; ++b, ++i)
                {
                    uu_[i] = uu_ptr_[b] * scaling;
                    vv_[i] = vv_ptr_[b] * scaling;
                    ww_[i] = ww_ptr_[b] * scaling;
                }
            }
            else
            {
                float *uu_, *vv_, *ww_;
                const float *uu_ptr_, *vv_ptr_, *ww_ptr_;
                uu_ = oskar_mem_float(uu, &err);
                vv_ = oskar_mem_float(vv, &err);
                ww_ = oskar_mem_float(ww, &err);
                uu_ptr_ = oskar_mem_float_const(uu_ptr, &err);
                vv_ptr_ = oskar_mem_float_const(vv_ptr, &err);
                ww_ptr_ = oskar_mem_float_const(ww_ptr, &err);
                for (int b = 0; b < num_baselines; ++b, ++i)
                {
                    uu_[i] = uu_ptr_[b] * scaling;
                    vv_[i] = vv_ptr_[b] * scaling;
                    ww_[i] = ww_ptr_[b] * scaling;
                }
            }
        }
    }

    /* ====================================== TIME SYNTHESIS, FREQ SNAPSHOTS */
    else if (!settings->time_snapshots && settings->channel_snapshots)
    {
        for (int t = vis_time_range[0], i = 0; t <= vis_time_range[1]; ++t)
        {
            int coord_offset = t * num_baselines;
            oskar_mem_set_alias(uu_ptr, vis_uu, coord_offset, num_vis_coords, &err);
            oskar_mem_set_alias(vv_ptr, vis_vv, coord_offset, num_vis_coords, &err);
            oskar_mem_set_alias(ww_ptr, vis_ww, coord_offset, num_vis_coords, &err);
            if (err) return err;

            if (type == OSKAR_DOUBLE)
            {
                double *uu_, *vv_, *ww_;
                const double *uu_ptr_, *vv_ptr_, *ww_ptr_;
                uu_ = oskar_mem_double(uu, &err);
                vv_ = oskar_mem_double(vv, &err);
                ww_ = oskar_mem_double(ww, &err);
                uu_ptr_ = oskar_mem_double_const(uu_ptr, &err);
                vv_ptr_ = oskar_mem_double_const(vv_ptr, &err);
                ww_ptr_ = oskar_mem_double_const(ww_ptr, &err);
                for (int b = 0; b < num_baselines; ++b, ++i)
                {
                    uu_[i] = uu_ptr_[b];
                    vv_[i] = vv_ptr_[b];
                    ww_[i] = ww_ptr_[b];
                }
            }
            else
            {
                float *uu_, *vv_, *ww_;
                const float *uu_ptr_, *vv_ptr_, *ww_ptr_;
                uu_ = oskar_mem_float(uu, &err);
                vv_ = oskar_mem_float(vv, &err);
                ww_ = oskar_mem_float(ww, &err);
                uu_ptr_ = oskar_mem_float_const(uu_ptr, &err);
                vv_ptr_ = oskar_mem_float_const(vv_ptr, &err);
                ww_ptr_ = oskar_mem_float_const(ww_ptr, &err);
                for (int b = 0; b < num_baselines; ++b, ++i)
                {
                    uu_[i] = uu_ptr_[b];
                    vv_[i] = vv_ptr_[b];
                    ww_[i] = ww_ptr_[b];
                }
            }
        }
    }

    /* ====================================== TIME SYNTHESIS, FREQ SYNTHESIS */
    else
    {
        for (int c = vis_chan_range[0], i = 0; c <= vis_chan_range[1]; ++c)
        {
            double freq = freq_start_hz + c * freq_inc_hz;
            double scaling = freq/im_freq;

            for (int t = vis_time_range[0]; t <= vis_time_range[1]; ++t)
            {
                int coord_offset = t * num_baselines;
                oskar_mem_set_alias(uu_ptr, vis_uu, coord_offset, num_vis_coords, &err);
                oskar_mem_set_alias(vv_ptr, vis_vv, coord_offset, num_vis_coords, &err);
                oskar_mem_set_alias(ww_ptr, vis_ww, coord_offset, num_vis_coords, &err);
                if (err) return err;

                if (type == OSKAR_DOUBLE)
                {
                    double *uu_, *vv_, *ww_;
                    const double *uu_ptr_, *vv_ptr_, *ww_ptr_;
                    uu_ = oskar_mem_double(uu, &err);
                    vv_ = oskar_mem_double(vv, &err);
                    ww_ = oskar_mem_double(ww, &err);
                    uu_ptr_ = oskar_mem_double_const(uu_ptr, &err);
                    vv_ptr_ = oskar_mem_double_const(vv_ptr, &err);
                    ww_ptr_ = oskar_mem_double_const(ww_ptr, &err);
                    for (int b = 0; b < num_baselines; ++b, ++i)
                    {
                        uu_[i] = uu_ptr_[b] * scaling;
                        vv_[i] = vv_ptr_[b] * scaling;
                        ww_[i] = ww_ptr_[b] * scaling;
                    }
                }
                else
                {
                    float *uu_, *vv_, *ww_;
                    const float *uu_ptr_, *vv_ptr_, *ww_ptr_;
                    uu_ = oskar_mem_float(uu, &err);
                    vv_ = oskar_mem_float(vv, &err);
                    ww_ = oskar_mem_float(ww, &err);
                    uu_ptr_ = oskar_mem_float_const(uu_ptr, &err);
                    vv_ptr_ = oskar_mem_float_const(vv_ptr, &err);
                    ww_ptr_ = oskar_mem_float_const(ww_ptr, &err);
                    for (int b = 0; b < num_baselines; ++b, ++i)
                    {
                        uu_[i] = uu_ptr_[b] * scaling;
                        vv_[i] = vv_ptr_[b] * scaling;
                        ww_[i] = ww_ptr_[b] * scaling;
                    }
                }
            }
        }
    }

    oskar_mem_free(uu_ptr, &err);
    oskar_mem_free(vv_ptr, &err);
    oskar_mem_free(ww_ptr, &err);

    return err;
}

#ifdef __cplusplus
}
#endif
