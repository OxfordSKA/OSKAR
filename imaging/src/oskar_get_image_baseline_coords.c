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

#ifdef __cplusplus
extern "C" {
#endif

int oskar_get_image_baseline_coords(oskar_Mem* uu, oskar_Mem* vv,
        const oskar_Visibilities* vis, int time, int channel,
        const oskar_SettingsImage* settings)
{
    int coord_offset;
    int num_vis;
    int chan_range[2];
    int time_range[2];
    double im_freq_start_hz;
    int location, type;
    int i, j, k, idx;
    double freq, scaling;
    oskar_Mem uu_ptr, vv_ptr;
    int err;

    /* Setup local variables */
    err = OSKAR_SUCCESS;
    num_vis = uu->num_elements;
    chan_range[0] = settings->channel_range[0];
    chan_range[1] = settings->channel_range[1];
    time_range[0] = settings->time_range[0];
    time_range[1] = settings->channel_range[1];
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

    /* == TIME SNAPSHOTS, FREQ SNAPSHOTS == */
    if (settings->time_snapshots && settings->channel_snapshots)
    {
        coord_offset = time * vis->num_baselines;
        err = oskar_mem_get_pointer(uu, &vis->uu_metres, coord_offset, num_vis);
        if (err) return err;
        err = oskar_mem_get_pointer(vv, &vis->vv_metres, coord_offset, num_vis);
        if (err) return err;
    }


    /* == TIME SNAPSHOTS, FREQ SYNTHESIS == */
    else if (settings->time_snapshots && !settings->channel_snapshots)
    {
        /* freq0 = mid freq of the imaging band. i.e. the freq we are imaging at */
        coord_offset = time * vis->num_baselines;
        err = oskar_mem_get_pointer(&uu_ptr, &vis->uu_metres, coord_offset, num_vis);
        if (err) return err;
        err = oskar_mem_get_pointer(&vv_ptr, &vis->vv_metres, coord_offset, num_vis);
        if (err) return err;
        for (j = 0; j < (chan_range[1] - chan_range[0] + 1); ++j)
        {
            freq = vis->freq_start_hz + ((chan_range[0] + j) * vis->freq_inc_hz);
            scaling = freq/im_freq_start_hz;
            for (i = 0; i < vis->num_baselines; ++i)
            {
                idx = j*vis->num_baselines + i;
                if (type == OSKAR_DOUBLE)
                {
                    ((double*)uu->data)[idx] = ((double*)uu_ptr.data)[i] * scaling;
                    ((double*)vv->data)[idx] = ((double*)vv_ptr.data)[i] * scaling;
                }
                else
                {
                    ((float*)uu->data)[idx] = ((float*)uu_ptr.data)[i] * scaling;
                    ((float*)vv->data)[idx] = ((float*)vv_ptr.data)[i] * scaling;
                }
            }
        }
    }

    /* == TIME SYNTHESIS, FREQ SNAPSHOTS == */
    else if (!settings->time_snapshots && settings->channel_snapshots)
    {
        /* TODO */
    }

    /* == TIME SYNTHESIS, FREQ SYNTHESIS == */
    else
    {
        int num_times_synth = (time_range[1]-time_range[0]+1);
        for (j = 0; j < (chan_range[1] - chan_range[0] + 1); ++j)
        {
            freq = vis->freq_start_hz + ((chan_range[0] + j) * vis->freq_inc_hz);
            scaling = freq/im_freq_start_hz;
            for (k = 0; k < num_times_synth; ++k)
            {
                coord_offset = (time_range[0] + k) * vis->num_baselines;
                err = oskar_mem_get_pointer(&uu_ptr, &vis->uu_metres, coord_offset, num_vis);
                if (err) return err;
                err = oskar_mem_get_pointer(&vv_ptr, &vis->vv_metres, coord_offset, num_vis);
                if (err) return err;
                for (i = 0; i < vis->num_baselines; ++i)
                {
                    idx = (j * num_times_synth + k) * vis->num_baselines + i;
                    if (type == OSKAR_DOUBLE)
                    {
                        ((double*)uu->data)[idx] = ((double*)uu_ptr.data)[i] * scaling;
                        ((double*)vv->data)[idx] = ((double*)vv_ptr.data)[i] * scaling;
                    }
                    else
                    {
                        ((float*)uu->data)[idx] = ((float*)uu_ptr.data)[i] * scaling;
                        ((float*)vv->data)[idx] = ((float*)vv_ptr.data)[i] * scaling;
                    }
                }
            }
        }
    }

    /* Cleanup */
    err = oskar_mem_free(&uu_ptr);
    if (err) return err;
    err = oskar_mem_free(&vv_ptr);
    if (err) return err;

    return OSKAR_SUCCESS;
}


#ifdef __cplusplus
}
#endif
