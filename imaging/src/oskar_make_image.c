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


#include "imaging/oskar_make_image.h"

#include "imaging/oskar_make_image_dft.h"
#include "imaging/oskar_image_resize.h"
#include "imaging/oskar_evaluate_image_lm_grid.h"
#include "imaging/oskar_get_image_baseline_coords.h"
#include "imaging/oskar_get_image_stokes.h"
#include "imaging/oskar_setup_image.h"

#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_mem_get_pointer.h"
#include "utility/oskar_mem_copy.h"
#include "utility/oskar_mem_assign.h"
#include "utility/oskar_mem_copy.h"
#include "utility/oskar_vector_types.h"


#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define SEC2DAYS 1.15740740740740740740741e-5

#ifdef __cplusplus
extern "C" {
#endif

int oskar_make_image(oskar_Image* im, const oskar_Visibilities* vis,
        const oskar_SettingsImage* settings)
{
    oskar_Mem l, m, stokes, uu, vv, amp, im_slice;
    int t, c, p, i, j, k; /* loop indices */
    int type;
    int size, num_pixels, location, num_pols, num_times, num_chan; /* dims */
    int pol_type;
    int time_range[2], chan_range[2];
    int num_vis_pols;
    int num_vis; /* number of visibilities passed to image per plane of the cube */
    double fov, freq0;
    int slice_offset;
    int err;
    int idx;

    /* Set the location for temporary memory used in this function */
    location = OSKAR_LOCATION_CPU;

    /* ___ Set local variables ___ */
    /* data type */
    if (im == NULL || vis == NULL || settings == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;


    type = (oskar_mem_is_double(vis->amplitude.type) &&
            oskar_mem_is_double(im->data.type)) ? OSKAR_DOUBLE : OSKAR_SINGLE;

    /* image variables */
    size = settings->size;
    fov = settings->fov_deg * M_PI/180.0;
    time_range[0] = settings->time_range[0];
    time_range[1] = settings->time_range[1];
    chan_range[0] = settings->channel_range[0];
    chan_range[1] = settings->channel_range[1];
    if (time_range[1] > vis->num_times-1) return OSKAR_ERR_OUT_OF_RANGE;
    if (time_range[0] < 0) time_range[0] = 0;
    if (time_range[1] < 0) time_range[1] = (settings->time_snapshots) ?
            vis->num_times-1 : 0;
    if (chan_range[1] > vis->num_channels-1) return OSKAR_ERR_OUT_OF_RANGE;
    if (chan_range[0] < 0) chan_range[0] = 0;
    if (chan_range[1] < 0) chan_range[1] = (settings->channel_snapshots) ?
            vis->num_channels-1 : 0;
    num_pixels = size*size;
    num_times = (settings->time_snapshots) ?
            (time_range[1] - time_range[0] + 1) : 1;
    if (num_times < 1) return OSKAR_ERR_INVALID_RANGE;
    num_chan  = (settings->channel_snapshots) ?
            (chan_range[1] - chan_range[0] + 1) : 1;
    if (num_chan < 1) return OSKAR_ERR_INVALID_RANGE;
    pol_type = settings->polarisation;
    if (pol_type == OSKAR_IMAGE_TYPE_STOKES_I ||
            pol_type == OSKAR_IMAGE_TYPE_STOKES_Q ||
            pol_type == OSKAR_IMAGE_TYPE_STOKES_U ||
            pol_type == OSKAR_IMAGE_TYPE_STOKES_V ||
            pol_type == OSKAR_IMAGE_TYPE_POL_XX ||
            pol_type == OSKAR_IMAGE_TYPE_POL_YY ||
            pol_type == OSKAR_IMAGE_TYPE_POL_XY ||
            pol_type == OSKAR_IMAGE_TYPE_POL_YX)
    {
        num_pols = 1;
    }
    else if (pol_type == OSKAR_IMAGE_TYPE_STOKES ||
            pol_type == OSKAR_IMAGE_TYPE_POL_LINEAR)
    {
        num_pols = 4;
    }
    else
        return OSKAR_ERR_BAD_DATA_TYPE;

    /* visibility variables */
    num_vis_pols = oskar_mem_is_matrix(vis->amplitude.type) ? 4 : 1;
    /* sanity checks */
    if (num_times > vis->num_times || num_chan > vis->num_channels ||
            num_pols > num_vis_pols)
    {
        return OSKAR_ERR_DIMENSION_MISMATCH;
    }
    if (num_vis_pols == 1 && pol_type != OSKAR_IMAGE_TYPE_STOKES_I)
        return OSKAR_ERR_SETTINGS;

    /* ___ Evaluate IQUV if required ___ */
    oskar_mem_init(&stokes, type, location, 0, OSKAR_FALSE);
    err = oskar_get_image_stokes(&stokes, vis, settings);
    if (err) return err;

    /* Setup the image */
    oskar_setup_image(im, vis, settings);

    /* Note: vis are channel -> time -> baseline order currently  */
    /*       vis coordinates are of length = num_times * num_baselines */
    /*       vis amp is of length = num_channels * num_times * num_baselines */
    if (settings->time_snapshots && settings->channel_snapshots)
    {
        num_vis = vis->num_baselines;
        oskar_mem_init(&uu,  type, location, num_vis, OSKAR_FALSE);
        oskar_mem_init(&vv,  type, location, num_vis, OSKAR_FALSE);
        oskar_mem_init(&amp, type | OSKAR_COMPLEX, location, num_vis, OSKAR_TRUE);
    }
    else if (settings->time_snapshots && !settings->channel_snapshots)
    {
        num_vis = vis->num_baselines * vis->num_channels;
        oskar_mem_init(&uu,  type, location, num_vis, OSKAR_TRUE);
        oskar_mem_init(&vv,  type, location, num_vis, OSKAR_TRUE);
        oskar_mem_init(&amp, type | OSKAR_COMPLEX, location, num_vis, OSKAR_TRUE);
    }
    else if (!settings->time_snapshots && settings->channel_snapshots)
    {
        num_vis = vis->num_baselines * vis->num_times;
        oskar_mem_init(&uu,  type, location, num_vis, OSKAR_TRUE);
        oskar_mem_init(&vv,  type, location, num_vis, OSKAR_TRUE);
        oskar_mem_init(&amp, type | OSKAR_COMPLEX, location, num_vis, OSKAR_TRUE);
        return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    }
    else /* freq and time synth */
    {
        num_vis = vis->num_baselines * vis->num_channels * vis->num_times;
        oskar_mem_init(&uu,  type, location, num_vis, OSKAR_TRUE);
        oskar_mem_init(&vv,  type, location, num_vis, OSKAR_TRUE);
        oskar_mem_init(&amp, type | OSKAR_COMPLEX, location, num_vis, OSKAR_TRUE);
    }

    oskar_mem_init(&im_slice, type, location, num_pixels, OSKAR_FALSE);


    /* ___ DFT: Allocate pixel grid ___ */
    if (settings->dft)
    {
        /* Generate lm grid. */
        oskar_mem_init(&l, type, location, num_pixels, OSKAR_TRUE);
        oskar_mem_init(&m, type, location, num_pixels, OSKAR_TRUE);
        if (type == OSKAR_SINGLE)
        {
            oskar_evaluate_image_lm_grid_f(size, size, fov, fov, (float*)l.data,
                    (float*)m.data);
        }
        else
        {
            oskar_evaluate_image_lm_grid_d(size, size, fov, fov, (double*)l.data,
                    (double*)m.data);
        }
    }
















    /* ___ Make the image ___ */
    for (c = 0; c < num_chan; ++c)
    {
        int channel = chan_range[0] + c;
        freq0 = im->freq_start_hz + (im->freq_inc_hz) * c + vis->channel_bandwidth_hz/2.0;

        for (t = 0; t < num_times; ++t)
        {
            int time = time_range[0] + t;

            /* Evaluate baseline coords needed for imaging */
            err = oskar_get_image_baseline_coords(&uu, &vv, vis, time, channel, settings);
            if (err) return err;

            for (p = 0; p < num_pols; ++p)
            {
                /* ___ Get visibility amplitudes for imaging ___ */
                /* Snapshots in both frequency and time */
                if (settings->time_snapshots && settings->channel_snapshots)
                {
                    if (num_pols == 4)
                    {
                        if (pol_type == OSKAR_IMAGE_TYPE_STOKES)
                        {
                            if (type == OSKAR_DOUBLE)
                            {
                                double4c* s_ = (double4c*)stokes.data;
                                double2* a_  = (double2*)amp.data;
                                int offset = (channel * vis->num_times + time)*vis->num_baselines;
                                for (j = 0; j < vis->num_baselines; ++j)
                                {
                                    if (p == 0)
                                    {
                                        a_[j].x = s_[offset + j].a.x;
                                        a_[j].y = s_[offset + j].a.y;
                                    }
                                    else if (p == 1)
                                    {
                                        a_[j].x = s_[offset + j].b.x;
                                        a_[j].y = s_[offset + j].b.y;
                                    }
                                    else if (p == 2)
                                    {
                                        a_[j].x = s_[offset + j].c.x;
                                        a_[j].y = s_[offset + j].c.y;
                                    }
                                    else if (p == 3)
                                    {
                                        a_[j].x = s_[offset + j].d.x;
                                        a_[j].y = s_[offset + j].d.y;
                                    }
                                }
                            }
                            else /* type == OSKAR_SINGLE */
                            {
                                float4c* s_ = (float4c*)stokes.data;
                                float2* a_  = (float2*)amp.data;
                                int offset = (channel * vis->num_times + time) * vis->num_baselines;
                                for (j = 0; j < vis->num_baselines; ++j)
                                {
                                    if (p == 0)
                                    {
                                        a_[j].x = s_[offset + j].a.x;
                                        a_[j].y = s_[offset + j].a.y;
                                    }
                                    else if (p == 1)
                                    {
                                        a_[j].x = s_[offset + j].b.x;
                                        a_[j].y = s_[offset + j].b.y;
                                    }
                                    else if (p == 2)
                                    {
                                        a_[j].x = s_[offset + j].c.x;
                                        a_[j].y = s_[offset + j].c.y;
                                    }
                                    else if (p == 3)
                                    {
                                        a_[j].x = s_[offset + j].d.x;
                                        a_[j].y = s_[offset + j].d.y;
                                    }
                                }
                            } /* [end] else typ == SINGLE */
                        } /* if (pol_type == STOKES) */
                        else if (pol_type == OSKAR_IMAGE_TYPE_POL_LINEAR)
                        {
                            if (type == OSKAR_DOUBLE)
                            {
                                double4c* v_ = (double4c*)vis->amplitude.data;
                                double2* a_ = (double2*)amp.data;
                                idx = (channel * vis->num_times + time) *
                                        vis->num_baselines;

                                for (j = 0; j < vis->num_baselines; ++j)
                                {
                                    if (p == 0)
                                    {
                                        a_[j].x = v_[idx + j].a.x;
                                        a_[j].y = v_[idx + j].a.y;
                                    }
                                    else if (p == 1)
                                    {
                                        a_[j].x = v_[idx + j].b.x;
                                        a_[j].y = v_[idx + j].b.y;
                                    }
                                    else if (p == 2)
                                    {
                                        a_[j].x = v_[idx + j].c.x;
                                        a_[j].y = v_[idx + j].c.y;
                                    }
                                    else if (p == 3)
                                    {
                                        a_[j].x = v_[idx + j].d.x;
                                        a_[j].y = v_[idx + j].d.y;
                                    }
                                }
                            }
                            else /* type == OSKAR_SINGLE */
                            {
                                float2* a_ = (float2*)amp.data;
                                float4c* v_ = (float4c*)vis->amplitude.data;
                                idx = (channel * vis->num_times + time) * vis->num_baselines;
                                for (j = 0; j < vis->num_baselines; ++j)
                                {
                                    if (p == 0)
                                    {
                                        a_[j].x = v_[idx + j].a.x;
                                        a_[j].y = v_[idx + j].a.y;
                                    }
                                    else if (p == 1)
                                    {
                                        a_[j].x = v_[idx + j].b.x;
                                        a_[j].y = v_[idx + j].b.y;
                                    }
                                    else if (p == 2)
                                    {
                                        a_[j].x = v_[idx + j].c.x;
                                        a_[j].y = v_[idx + j].c.y;
                                    }
                                    else if (p == 3)
                                    {
                                        a_[j].x = v_[idx + j].d.x;
                                        a_[j].y = v_[idx + j].d.y;
                                    }
                                }
                            }
                        } /* else if (pol_type == LINEAR) */
                        else return OSKAR_ERR_DIMENSION_MISMATCH;
                    } /* if (num_pols == 4) */
                    else if (num_pols == 1)
                    {
                        if (pol_type == OSKAR_IMAGE_TYPE_STOKES_I ||
                                pol_type == OSKAR_IMAGE_TYPE_STOKES_Q ||
                                pol_type == OSKAR_IMAGE_TYPE_STOKES_U ||
                                pol_type == OSKAR_IMAGE_TYPE_STOKES_V)
                        {
                            if (type == OSKAR_DOUBLE)
                            {
                                double2* v_ = (double2*)stokes.data;
                                double2* a_ = (double2*)amp.data;
                                idx = (channel * vis->num_times + time) * vis->num_baselines;
                                for (j = 0; j < vis->num_baselines; ++j)
                                {
                                    a_[j].x = v_[idx + j].x;
                                    a_[j].y = v_[idx + j].y;
                                }
                            }
                            else /* type == OSKAR_SINGLE */
                            {
                                float2* v_ = (float2*)stokes.data;
                                float2* a_ = (float2*)amp.data;
                                idx = (channel * vis->num_times + time) * vis->num_baselines;
                                for (j = 0; j < vis->num_baselines; ++j)
                                {
                                    a_[j].x = v_[idx + j].x;
                                    a_[j].y = v_[idx + j].y;
                                }
                            }
                        } /* [end] if pol = I|Q|U|V */
                        else if (pol_type == OSKAR_IMAGE_TYPE_POL_XX ||
                                pol_type == OSKAR_IMAGE_TYPE_POL_XY ||
                                pol_type == OSKAR_IMAGE_TYPE_POL_YX ||
                                pol_type == OSKAR_IMAGE_TYPE_POL_YY)
                        {
                            if (type == OSKAR_DOUBLE)
                            {
                                double4c* v_ = (double4c*)vis->amplitude.data;
                                double2* a_ = (double2*)amp.data;
                                idx = (channel * vis->num_times + time) * vis->num_baselines;
                                for (j = 0; j < vis->num_baselines; ++j)
                                {
                                    if (pol_type == OSKAR_IMAGE_TYPE_POL_XX)
                                    {
                                        a_[j].x = v_[idx + j].a.x;
                                        a_[j].y = v_[idx + j].a.y;
                                    }
                                    else if (pol_type == OSKAR_IMAGE_TYPE_POL_XY)
                                    {
                                        a_[j].x = v_[idx + j].b.x;
                                        a_[j].y = v_[idx + j].b.y;
                                    }
                                    else if (pol_type == OSKAR_IMAGE_TYPE_POL_YX)
                                    {
                                        a_[j].x = v_[idx + j].c.x;
                                        a_[j].y = v_[idx + j].c.y;
                                    }
                                    else if (pol_type == OSKAR_IMAGE_TYPE_POL_YY)
                                    {
                                        a_[j].x = v_[idx + j].d.x;
                                        a_[j].y = v_[idx + j].d.y;
                                    }
                                }
                            }
                            else /* type == OSKAR_SINGLE */
                            {
                                float4c* v_ = (float4c*)vis->amplitude.data;
                                float2* a_ = (float2*)amp.data;
                                idx = (channel * vis->num_times + time) * vis->num_baselines;
                                for (j = 0; j < vis->num_baselines; ++j)
                                {
                                    if (pol_type == OSKAR_IMAGE_TYPE_POL_XX)
                                    {
                                        a_[j].x = v_[idx + j].a.x;
                                        a_[j].y = v_[idx + j].a.y;
                                    }
                                    else if (pol_type == OSKAR_IMAGE_TYPE_POL_XY)
                                    {
                                        a_[j].x = v_[idx + j].b.x;
                                        a_[j].y = v_[idx + j].b.y;
                                    }
                                    else if (pol_type == OSKAR_IMAGE_TYPE_POL_YX)
                                    {
                                        a_[j].x = v_[idx + j].c.x;
                                        a_[j].y = v_[idx + j].c.y;
                                    }
                                    else if (pol_type == OSKAR_IMAGE_TYPE_POL_YY)
                                    {
                                        a_[j].x = v_[idx + j].d.x;
                                        a_[j].y = v_[idx + j].d.y;
                                    }
                                }
                            }
                        }
                    }
                    else return OSKAR_ERR_DIMENSION_MISMATCH;
                }


                /* Evaluate amps (snapshots in time, frequency synthesis) */
                else if (settings->time_snapshots && !settings->channel_snapshots)
                {
                    if (num_pols == 4)
                    {
                        if (pol_type == OSKAR_IMAGE_TYPE_STOKES)
                        {
                            if (type == OSKAR_DOUBLE)
                            {
                                double4c* v_ = (double4c*)stokes.data;
                                double2* a_ = (double2*)amp.data;
                                /* Concatenate amplitudes from the frequencies */
                                for (k = 0, j = chan_range[0]; j <= chan_range[1]; ++j)
                                {
                                    for (i = 0; i < vis->num_baselines; ++i, ++k)
                                    {
                                        idx = (j*vis->num_times+time)*vis->num_baselines+i;
                                        if (p == 0)
                                        {
                                            a_[k].x = v_[idx].a.x;
                                            a_[k].y = v_[idx].a.y;
                                        }
                                        else if (p == 1)
                                        {
                                            a_[k].x = v_[idx].b.x;
                                            a_[k].y = v_[idx].b.y;
                                        }
                                        else if (p == 2)
                                        {
                                            a_[k].x = v_[idx].c.x;
                                            a_[k].y = v_[idx].c.y;
                                        }
                                        else /* (p == 3) */
                                        {
                                            a_[k].x = v_[idx].d.x;
                                            a_[k].y = v_[idx].d.y;
                                        }
                                    }
                                }
                            }
                            else /* type == SINGLE */
                            {
                                float4c* v_ = (float4c*)stokes.data;
                                float2* a_ = (float2*)amp.data;
                                /* Concatenate amplitudes from the frequencies */
                                for (k = 0, j = chan_range[0]; j <= chan_range[1]; ++j)
                                {
                                    for (i = 0; i < vis->num_baselines; ++i, ++k)
                                    {
                                        idx = (j*vis->num_times+time)*vis->num_baselines+i;
                                        if (p == 0)
                                        {
                                            a_[k].x = v_[idx].a.x;
                                            a_[k].y = v_[idx].a.y;
                                        }
                                        else if (p == 1)
                                        {
                                            a_[k].x = v_[idx].b.x;
                                            a_[k].y = v_[idx].b.y;
                                        }
                                        else if (p == 2)
                                        {
                                            a_[k].x = v_[idx].c.x;
                                            a_[k].y = v_[idx].c.y;
                                        }
                                        else /* (p == 3) */
                                        {
                                            a_[k].x = v_[idx].d.x;
                                            a_[k].y = v_[idx].d.y;
                                        }
                                    }
                                }
                            }
                        }
                        else if (pol_type == OSKAR_IMAGE_TYPE_POL_LINEAR)
                        {
                            /* Time snapshots, freq synth, linear (4pol), double */
                            if (type == OSKAR_DOUBLE)
                            {
                                double4c* v_ = (double4c*)vis->amplitude.data;
                                double2* a_ = (double2*)amp.data;
                                /* Concatenate amplitudes from the frequencies */
                                for (k = 0, j = chan_range[0]; j <= chan_range[1]; ++j)
                                {
                                    for (i = 0; i < vis->num_baselines; ++i, ++k)
                                    {
                                        idx = (j*vis->num_times+time)*vis->num_baselines+i;
                                        if (p == 0)
                                        {
                                            a_[k].x = v_[idx].a.x;
                                            a_[k].y = v_[idx].a.y;
                                        }
                                        else if (p == 1)
                                        {
                                            a_[k].x = v_[idx].b.x;
                                            a_[k].y = v_[idx].b.y;
                                        }
                                        else if (p == 2)
                                        {
                                            a_[k].x = v_[idx].c.x;
                                            a_[k].y = v_[idx].c.y;
                                        }
                                        else /* (p == 3) */
                                        {
                                            a_[k].x = v_[idx].d.x;
                                            a_[k].y = v_[idx].d.y;
                                        }
                                    }
                                }
                            }
                            else /* type == SINGLE */
                            {
                                float4c* v_ = (float4c*)vis->amplitude.data;
                                float2* a_ = (float2*)amp.data;
                                /* Concatenate amplitudes from the frequencies */
                                for (k = 0, j = chan_range[0]; j <= chan_range[1]; ++j)
                                {
                                    for (i = 0; i < vis->num_baselines; ++i, ++k)
                                    {
                                        idx = (j*vis->num_times+time)*vis->num_baselines+i;
                                        if (p == 0)
                                        {
                                            a_[k].x = v_[idx].a.x;
                                            a_[k].y = v_[idx].a.y;
                                        }
                                        else if (p == 1)
                                        {
                                            a_[k].x = v_[idx].b.x;
                                            a_[k].y = v_[idx].b.y;
                                        }
                                        else if (p == 2)
                                        {
                                            a_[k].x = v_[idx].c.x;
                                            a_[k].y = v_[idx].c.y;
                                        }
                                        else /* (p == 3) */
                                        {
                                            a_[k].x = v_[idx].d.x;
                                            a_[k].y = v_[idx].d.y;
                                        }
                                    }
                                }
                            } /* [end] else type == SINGLE */
                        } /* [end] if linear */
                    } /* [end] if (num_pols == 4) */
                    else if (num_pols == 1)
                    {
                        if (pol_type == OSKAR_IMAGE_TYPE_STOKES_I ||
                                pol_type == OSKAR_IMAGE_TYPE_STOKES_Q ||
                                pol_type == OSKAR_IMAGE_TYPE_STOKES_U ||
                                pol_type == OSKAR_IMAGE_TYPE_STOKES_V)
                        {
                            if (type == OSKAR_DOUBLE)
                            {
                                double2* v_ = (double2*)stokes.data;
                                double2* a_ = (double2*)amp.data;
                                /* Concatenate amplitudes from the frequencies */
                                for (k = 0, j = chan_range[0]; j <= chan_range[1]; ++j)
                                {
                                    for (i = 0; i < vis->num_baselines; ++i, ++k)
                                    {
                                        idx = (j*vis->num_times+time)*vis->num_baselines+i;
                                        a_[k].x = v_[idx].x;
                                        a_[k].y = v_[idx].y;
                                    }
                                }
                            }
                            else /* type == SINGLE */
                            {
                                float2* v_ = (float2*)stokes.data;
                                float2* a_ = (float2*)amp.data;
                                /* Concatenate amplitudes from the frequencies */
                                for (k = 0, j = chan_range[0]; j <= chan_range[1]; ++j)
                                {
                                    for (i = 0; i < vis->num_baselines; ++i, ++k)
                                    {
                                        idx = (j*vis->num_times+time)*vis->num_baselines+i;
                                        a_[k].x = v_[idx].x;
                                        a_[k].y = v_[idx].y;
                                    }
                                }
                            }
                        }
                        else if (pol_type == OSKAR_IMAGE_TYPE_POL_XX ||
                                pol_type == OSKAR_IMAGE_TYPE_POL_XY ||
                                pol_type == OSKAR_IMAGE_TYPE_POL_YX ||
                                pol_type == OSKAR_IMAGE_TYPE_POL_YY)
                        {
                            if (type == OSKAR_DOUBLE)
                            {
                                /* time snapshots, freq synth, XX|XY|YX|YY (1pol), double */
                                double4c* v_ = (double4c*)vis->amplitude.data;
                                double2* a_ = (double2*)amp.data;
                                /* Concatenate amplitudes from the frequencies */
                                for (k = 0, j = chan_range[0]; j <= chan_range[1]; ++j)
                                {
                                    for (i = 0; i < vis->num_baselines; ++i, ++k)
                                    {
                                        idx = (j*vis->num_times+time)*vis->num_baselines+i;
                                        if (pol_type == OSKAR_IMAGE_TYPE_POL_XX)
                                        {
                                            a_[k].x = v_[idx].a.x;
                                            a_[k].y = v_[idx].a.y;
                                        }
                                        else if (pol_type == OSKAR_IMAGE_TYPE_POL_XY)
                                        {
                                            a_[k].x = v_[idx].b.x;
                                            a_[k].y = v_[idx].b.y;
                                        }
                                        else if (pol_type == OSKAR_IMAGE_TYPE_POL_YX)
                                        {
                                            a_[k].x = v_[idx].c.x;
                                            a_[k].y = v_[idx].c.y;
                                        }
                                        else /* pol_type == OSKAR_IMAGE_TYPE_POL_YY */
                                        {
                                            a_[k].x = v_[idx].d.x;
                                            a_[k].y = v_[idx].d.y;
                                        }
                                    }
                                }
                            }
                            else /* type == SINGLE */
                            {
                                /* Time snapshots, freq synth, XX|XY|YX|YY (1pol), single */
                                float4c* v_ = (float4c*)vis->amplitude.data;
                                float2* a_ = (float2*)amp.data;
                                /* Concatenate amplitudes from the frequencies */
                                for (k = 0, j = chan_range[0]; j <= chan_range[1]; ++j)
                                {
                                    for (i = 0; i < vis->num_baselines; ++i, ++k)
                                    {
                                        idx = (j*vis->num_times+time)*vis->num_baselines+i;
                                        if (pol_type == OSKAR_IMAGE_TYPE_POL_XX)
                                        {
                                            a_[k].x = v_[idx].a.x;
                                            a_[k].y = v_[idx].a.y;
                                        }
                                        else if (pol_type == OSKAR_IMAGE_TYPE_POL_XY)
                                        {
                                            a_[k].x = v_[idx].b.x;
                                            a_[k].y = v_[idx].b.y;
                                        }
                                        else if (pol_type == OSKAR_IMAGE_TYPE_POL_YX)
                                        {
                                            a_[k].x = v_[idx].c.x;
                                            a_[k].y = v_[idx].c.y;
                                        }
                                        else /* pol_type == OSKAR_IMAGE_TYPE_POL_YY */
                                        {
                                            a_[k].x = v_[idx].d.x;
                                            a_[k].y = v_[idx].d.y;
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            return OSKAR_ERR_DIMENSION_MISMATCH;
                        }
                    }
                }
                /* FIXME WRONG INDENT LEVEL ON THIS ELSE ... */
                else /* Evaluate amps (frequency and time synthesis) */
                {
                    if (num_pols == 4)
                    {
                        if (pol_type == OSKAR_IMAGE_TYPE_STOKES)
                        {
                            if (type == OSKAR_DOUBLE)
                            {
                                /* TODO time synth, freq synth, stokes (4pol), double */
                                return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
                            }
                            else /* type == SINGLE */
                            {
                                /* TODO */
                                return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
                            }
                        }
                        else if (pol_type == OSKAR_IMAGE_TYPE_POL_LINEAR)
                        {
                            if (type == OSKAR_DOUBLE)
                            {
                                /* TODO */
                                return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
                            }
                            else /* type == SINGLE */
                            {
                                /* TODO */
                                return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
                            }
                        }
                        else
                        {
                            return OSKAR_ERR_DIMENSION_MISMATCH;
                        }
                    } /* if (num_pols == 4) */
                    else if (num_pols == 1)
                    {
                        if (pol_type == OSKAR_IMAGE_TYPE_STOKES_I ||
                                pol_type == OSKAR_IMAGE_TYPE_STOKES_Q ||
                                pol_type == OSKAR_IMAGE_TYPE_STOKES_U ||
                                pol_type == OSKAR_IMAGE_TYPE_STOKES_V)
                        {
                            if (type == OSKAR_DOUBLE)
                            {
                                /* TODO */
                                return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
                            }
                            else /* type == SINGLE */
                            {
                                /* TODO */
                                return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
                            }
                        }
                        else if (pol_type == OSKAR_IMAGE_TYPE_POL_XX ||
                                pol_type == OSKAR_IMAGE_TYPE_POL_XY ||
                                pol_type == OSKAR_IMAGE_TYPE_POL_YX ||
                                pol_type == OSKAR_IMAGE_TYPE_POL_YY)
                        {
                            if (type == OSKAR_DOUBLE)
                            {
                                /* TODO */
                                return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
                            }
                            else /* type == SINGLE */
                            {
                                /* TODO */
                                return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
                            }
                        }
                        else
                        {
                            return OSKAR_ERR_DIMENSION_MISMATCH;
                        }
                    }
                }

                /* Get a pointer to the slice of the image currently being
                 * imaged */
                slice_offset = c * num_times * num_pols * num_pixels +
                        t * num_pols * num_pixels + p * num_pixels;
                oskar_mem_get_pointer(&im_slice, &im->data,
                        slice_offset, num_pixels);

                /* ___ make the image ___ */
                if (settings->dft)
                {
                    err = oskar_make_image_dft(&im_slice, &uu, &vv, &amp,
                            &l, &m, freq0);
                    if (err) return err;
                }
                else
                {
                    /* FFT */
                    return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
                }

            } /* loop over image polarisation */
        } /* loop over image time */
    } /* loop over image channels */

    /* Clean up */
    oskar_mem_free(&l);
    oskar_mem_free(&m);
    oskar_mem_free(&stokes);
    oskar_mem_free(&uu);
    oskar_mem_free(&vv);
    oskar_mem_free(&amp);
    oskar_mem_free(&im_slice);

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
