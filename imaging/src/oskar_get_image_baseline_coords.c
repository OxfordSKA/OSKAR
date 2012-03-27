/*
 * Copyright (c) 2011, The University of Oxford
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

#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_mem_get_pointer.h"
#include "utility/oskar_mem_free.h"

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif


int oskar_get_image_baseline_coords(oskar_Mem* uu, oskar_Mem* vv,
        const oskar_Visibilities* vis, int time,
        const oskar_SettingsImage* settings)
{
    int coord_offset;
    int num_vis_coords;
    int chan_range[2];
    int time_range[2];
    double im_freq_start_hz;
    int location, type;
    int i, j, t, k;
    double freq, scaling;
    oskar_Mem uu_ptr, vv_ptr;
    int err = OSKAR_SUCCESS;

    /* Setup local variables */
    err = OSKAR_SUCCESS;
    num_vis_coords = vis->num_baselines;
    chan_range[0] = settings->channel_range[0];
    chan_range[1] = settings->channel_range[1];
    time_range[0] = settings->time_range[0];
    time_range[1] = settings->time_range[1];
    if (time_range[1] > vis->num_times-1) return OSKAR_ERR_OUT_OF_RANGE;
    if (time_range[0] < 0) time_range[0] = 0;
    if (time_range[1] < 0) time_range[1] = vis->num_times-1;
    if (chan_range[1] > vis->num_channels-1) return OSKAR_ERR_OUT_OF_RANGE;
    if (chan_range[0] < 0) chan_range[0] = 0;
    if (chan_range[1] < 0) chan_range[1] = vis->num_channels-1;

    if (settings->channel_snapshots)
    {
        im_freq_start_hz = vis->freq_start_hz + (chan_range[0] * vis->freq_inc_hz);
    }
    else
    {
        im_freq_start_hz = vis->freq_start_hz + (chan_range[0] * vis->freq_inc_hz) +
                ((chan_range[1]-chan_range[0]) *  vis->freq_inc_hz) / 2.0;
    }
    location = OSKAR_LOCATION_CPU;
    type = oskar_mem_is_double(vis->amplitude.type) ? OSKAR_DOUBLE : OSKAR_SINGLE;
    err = oskar_mem_init(&uu_ptr, type, location, vis->num_baselines, OSKAR_FALSE);
    if (err) return err;
    err = oskar_mem_init(&vv_ptr, type, location, vis->num_baselines, OSKAR_FALSE);
    if (err) return err;

    /* ========================================= TIME SNAPSHOTS, FREQ SNAPSHOTS */
    if (settings->time_snapshots && settings->channel_snapshots)
    {
        coord_offset = time * vis->num_baselines;
        err = oskar_mem_get_pointer(uu, &vis->uu_metres, coord_offset, num_vis_coords);
        if (err) goto stop;
        err = oskar_mem_get_pointer(vv, &vis->vv_metres, coord_offset, num_vis_coords);
        if (err) goto stop;
    }


    /* ======================================== TIME SNAPSHOTS, FREQ SYNTHESIS */
    else if (settings->time_snapshots && !settings->channel_snapshots)
    {
        /* freq0 = mid freq of the imaging band. i.e. the freq we are imaging at */
        coord_offset = time * vis->num_baselines;
        err = oskar_mem_get_pointer(&uu_ptr, &vis->uu_metres, coord_offset, num_vis_coords);
        if (err) goto stop;
        err = oskar_mem_get_pointer(&vv_ptr, &vis->vv_metres, coord_offset, num_vis_coords);
        if (err) goto stop;
        for (k = 0, j = chan_range[0]; j <= chan_range[1]; ++j)
        {
            freq = vis->freq_start_hz + j * vis->freq_inc_hz;
            scaling = freq/im_freq_start_hz;
            for (i = 0; i < vis->num_baselines; ++i, ++k)
            {
                if (type == OSKAR_DOUBLE)
                {
                    ((double*)uu->data)[k] = ((double*)uu_ptr.data)[i] * scaling;
                    ((double*)vv->data)[k] = ((double*)vv_ptr.data)[i] * scaling;
                }
                else
                {
                    ((float*)uu->data)[k] = ((float*)uu_ptr.data)[i] * scaling;
                    ((float*)vv->data)[k] = ((float*)vv_ptr.data)[i] * scaling;
                }
            }
        }
    }

    /* ======================================== TIME SYNTHESIS, FREQ SNAPSHOTS */
    else if (!settings->time_snapshots && settings->channel_snapshots)
    {
        for (k = 0, t = time_range[0]; t <= time_range[1]; ++t)
        {
            coord_offset = t * vis->num_baselines;
            err = oskar_mem_get_pointer(&uu_ptr, &vis->uu_metres, coord_offset, num_vis_coords);
            if (err) goto stop;
            err = oskar_mem_get_pointer(&vv_ptr, &vis->vv_metres, coord_offset, num_vis_coords);
            if (err) goto stop;
            for (i = 0; i < vis->num_baselines; ++i, ++k)
            {
                if (type == OSKAR_DOUBLE)
                {
                    ((double*)uu->data)[k] = ((double*)uu_ptr.data)[i];
                    ((double*)vv->data)[k] = ((double*)vv_ptr.data)[i];
                }
                else
                {
                    ((float*)uu->data)[k] = ((float*)uu_ptr.data)[i];
                    ((float*)vv->data)[k] = ((float*)vv_ptr.data)[i];
                }
            }
        }
    }

    /* ======================================== TIME SYNTHESIS, FREQ SYNTHESIS */
    else
    {
        for (k = 0, j = chan_range[0]; j <= chan_range[1]; ++j)
        {
            freq = vis->freq_start_hz + (j * vis->freq_inc_hz);
            scaling = freq/im_freq_start_hz;
            for (t = time_range[0]; t <= time_range[1]; ++t)
            {
                coord_offset = t * vis->num_baselines;
                err = oskar_mem_get_pointer(&uu_ptr, &vis->uu_metres, coord_offset, num_vis_coords);
                if (err) goto stop;
                err = oskar_mem_get_pointer(&vv_ptr, &vis->vv_metres, coord_offset, num_vis_coords);
                if (err) goto stop;
                for (i = 0; i < vis->num_baselines; ++i, ++k)
                {
                    if (type == OSKAR_DOUBLE)
                    {
                        ((double*)uu->data)[k] = ((double*)uu_ptr.data)[i] * scaling;
                        ((double*)vv->data)[k] = ((double*)vv_ptr.data)[i] * scaling;
                    }
                    else
                    {
                        ((float*)uu->data)[k] = ((float*)uu_ptr.data)[i] * scaling;
                        ((float*)vv->data)[k] = ((float*)vv_ptr.data)[i] * scaling;
                    }
                }
            }
        }
    }

    /* Cleanup */
    stop:
    err = oskar_mem_free(&uu_ptr);
    err = oskar_mem_free(&vv_ptr);

    return err;
}


#ifdef __cplusplus
}
#endif
