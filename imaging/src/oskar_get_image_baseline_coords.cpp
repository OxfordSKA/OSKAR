/*
 * Copyright (c) 2012, The University of Oxford
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


#include "imaging/oskar_get_image_baseline_coords.h"
#include "imaging/oskar_image_evaluate_ranges.h"

#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_mem_get_pointer.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_element_size.h"

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
    int err = OSKAR_SUCCESS;
    int location = OSKAR_LOCATION_CPU;
    int num_vis_coords = num_baselines;

    // Data ranges for frequency and time synthesis.
    int vis_time_range[2];
    err = oskar_evaluate_image_data_range(vis_time_range, settings->time_range,
            num_times);
    if (err) return err;
    int vis_chan_range[2];
    err = oskar_evaluate_image_data_range(vis_chan_range, settings->channel_range,
            num_channels);
    if (err) return err;

    // Declare temporary pointers into visibility coordinate arrays.
    int type = vis_uu->type;
    oskar_Mem uu_ptr(type, location, num_baselines, OSKAR_FALSE);
    oskar_Mem vv_ptr(type, location, num_baselines, OSKAR_FALSE);
    oskar_Mem ww_ptr(type, location, num_baselines, OSKAR_FALSE);

    /* ========================================= TIME SNAPSHOTS, FREQ SNAPSHOTS */
    if (settings->time_snapshots && settings->channel_snapshots)
    {
        int coord_offset = vis_time * num_baselines;
        size_t element_size = oskar_mem_element_size(type);
        printf("coord_offset = %i\n", coord_offset);
        memcpy(uu->data, (void*)((char*)vis_uu->data + element_size * coord_offset), element_size * num_baselines);
        memcpy(vv->data, (void*)((char*)vis_vv->data + element_size * coord_offset), element_size * num_baselines);
        memcpy(ww->data, (void*)((char*)vis_ww->data + element_size * coord_offset), element_size * num_baselines);
    }

    /* ======================================== TIME SNAPSHOTS, FREQ SYNTHESIS */
    else if (settings->time_snapshots && !settings->channel_snapshots)
    {
        int coord_offset = vis_time * num_baselines;
        oskar_mem_get_pointer(&uu_ptr, vis_uu, coord_offset, num_vis_coords, &err);
        if (err) return err;
        oskar_mem_get_pointer(&vv_ptr, vis_vv, coord_offset, num_vis_coords, &err);
        if (err) return err;
        oskar_mem_get_pointer(&ww_ptr, vis_ww, coord_offset, num_vis_coords, &err);
        if (err) return err;

        for (int i = 0, c = vis_chan_range[0]; c <= vis_chan_range[1]; ++c)
        {
            double freq = freq_start_hz + c * freq_inc_hz;
            double scaling = freq/im_freq;
            for (int b = 0; b < num_baselines; ++b, ++i)
            {
                if (type == OSKAR_DOUBLE)
                {
                    ((double*)uu->data)[i] = ((double*)uu_ptr.data)[b] * scaling;
                    ((double*)vv->data)[i] = ((double*)vv_ptr.data)[b] * scaling;
                    ((double*)ww->data)[i] = ((double*)ww_ptr.data)[b] * scaling;
                }
                else
                {
                    ((float*)uu->data)[i] = ((float*)uu_ptr.data)[b] * scaling;
                    ((float*)vv->data)[i] = ((float*)vv_ptr.data)[b] * scaling;
                    ((float*)ww->data)[i] = ((float*)ww_ptr.data)[b] * scaling;
                }
            }
        }
    }

    /* ======================================== TIME SYNTHESIS, FREQ SNAPSHOTS */
    else if (!settings->time_snapshots && settings->channel_snapshots)
    {
        for (int i = 0, t = vis_time_range[0]; t <= vis_time_range[1]; ++t)
        {
            int coord_offset = t * num_baselines;
            oskar_mem_get_pointer(&uu_ptr, vis_uu, coord_offset, num_vis_coords, &err);
            if (err) return err;
            oskar_mem_get_pointer(&vv_ptr, vis_vv, coord_offset, num_vis_coords, &err);
            if (err) return err;
            oskar_mem_get_pointer(&ww_ptr, vis_ww, coord_offset, num_vis_coords, &err);
            if (err) return err;

            for (int b = 0; b < num_baselines; ++b, ++i)
            {
                if (type == OSKAR_DOUBLE)
                {
                    ((double*)uu->data)[i] = ((double*)uu_ptr.data)[b];
                    ((double*)vv->data)[i] = ((double*)vv_ptr.data)[b];
                    ((double*)ww->data)[i] = ((double*)ww_ptr.data)[b];
                }
                else
                {
                    ((float*)uu->data)[i] = ((float*)uu_ptr.data)[b];
                    ((float*)vv->data)[i] = ((float*)vv_ptr.data)[b];
                    ((float*)ww->data)[i] = ((float*)ww_ptr.data)[b];
                }
            }
        }
    }

    /* ======================================== TIME SYNTHESIS, FREQ SYNTHESIS */
    else
    {
        for (int i = 0, c = vis_chan_range[0]; c <= vis_chan_range[1]; ++c)
        {
            double freq = freq_start_hz + c * freq_inc_hz;
            double scaling = freq/im_freq;

            for (int t = vis_time_range[0]; t <= vis_time_range[1]; ++t)
            {
                int coord_offset = t * num_baselines;
                oskar_mem_get_pointer(&uu_ptr, vis_uu, coord_offset, num_vis_coords, &err);
                if (err) return err;
                oskar_mem_get_pointer(&vv_ptr, vis_vv, coord_offset, num_vis_coords, &err);
                if (err) return err;
                oskar_mem_get_pointer(&ww_ptr, vis_ww, coord_offset, num_vis_coords, &err);
                if (err) return err;

                for (int b = 0; b < num_baselines; ++b, ++i)
                {
                    if (type == OSKAR_DOUBLE)
                    {
                        ((double*)uu->data)[i] = ((double*)uu_ptr.data)[b] * scaling;
                        ((double*)vv->data)[i] = ((double*)vv_ptr.data)[b] * scaling;
                        ((double*)ww->data)[i] = ((double*)ww_ptr.data)[b] * scaling;
                    }
                    else
                    {
                        ((float*)uu->data)[i] = ((float*)uu_ptr.data)[b] * scaling;
                        ((float*)vv->data)[i] = ((float*)vv_ptr.data)[b] * scaling;
                        ((float*)ww->data)[i] = ((float*)ww_ptr.data)[b] * scaling;
                    }
                }
            }
        }
    }

    return err;
}


#ifdef __cplusplus
}
#endif
