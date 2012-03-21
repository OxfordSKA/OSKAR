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
#define C_0 299792458.0

#ifdef __cplusplus
extern "C" {
#endif

int oskar_make_image(oskar_Image* image, const oskar_Visibilities* vis,
        const oskar_SettingsImage* settings)
{
    oskar_Mem l, m, stokes, uu, vv, amp, uu_ptr, vv_ptr, im_slice;
    int t, c, p, i, j, k; /* loop indices */
    int type;
    int size, num_pixels, location, num_pols, num_times, num_chan; /* dims */
    int pol_type;
    int time_range[2], chan_range[2];
    int num_vis_amps, num_vis_pols;
    int num_vis; /* number of visibilities passed to image per plane of the cube */
    double fov, freq0, lambda0, lambda, scaling;
    int slice_offset;
    int err;

    /* Set the location for temporary memory used in this function */
    location = OSKAR_LOCATION_CPU;

    /* ___ Set local variables ___ */
    /* data type */
    if (image == NULL || vis == NULL || settings == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (oskar_mem_is_double(vis->amplitude.type) &&
            oskar_mem_is_double(image->data.type))
    {
        type = OSKAR_DOUBLE;
    }
    else if (oskar_mem_is_single(vis->amplitude.type) &&
            oskar_mem_is_single(image->data.type))
    {
        type = OSKAR_SINGLE;
    }
    else
        return OSKAR_ERR_BAD_DATA_TYPE;


    /* image variables*/
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
            (time_range[1] - chan_range[0] + 1) : 1;
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
    num_vis_amps = vis->num_baselines * vis->num_channels * vis->num_times;
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

    /* If the input data is polarised and a single stokes polarisation type is selected */
    if (num_vis_pols == 4 && (pol_type == OSKAR_IMAGE_TYPE_STOKES_I ||
            pol_type == OSKAR_IMAGE_TYPE_STOKES_Q ||
            pol_type == OSKAR_IMAGE_TYPE_STOKES_U ||
            pol_type == OSKAR_IMAGE_TYPE_STOKES_V))
    {
        printf("** evaluating (I | Q | U | V)\n");
        /* I = 0.5 (XX + YY) */
        switch (pol_type)
        {
            case OSKAR_IMAGE_TYPE_STOKES_I:
            {
                oskar_mem_init(&stokes, type | OSKAR_COMPLEX, location, num_vis_amps, OSKAR_TRUE);
                for (i = 0; i < num_vis_amps; ++i)
                {
                    if (type == OSKAR_DOUBLE)
                    {
                        double4c* d_ = (double4c*)vis->amplitude.data;
                        double2* s_ = ((double2*)stokes.data);
                        s_[i].x = 0.5 * (d_[i].a.x + d_[i].d.x);
                        s_[i].y = 0.5 * (d_[i].a.y + d_[i].d.y);
                    }
                    else
                    {
                        float4c* d_ = (float4c*)vis->amplitude.data;
                        float2* s_ = (float2*)stokes.data;
                        s_[i].x = 0.5 * (d_[i].a.x + d_[i].d.x);
                        s_[i].y = 0.5 * (d_[i].a.y + d_[i].d.y);
                    }
                }
                break;
            }
            case OSKAR_IMAGE_TYPE_STOKES_Q:
            {
                oskar_mem_init(&stokes, type | OSKAR_COMPLEX, location, num_vis_amps, OSKAR_TRUE);
                for (i = 0; i < num_vis_amps; ++i)
                {
                    if (type == OSKAR_DOUBLE)
                    {
                        double4c* d_ = (double4c*)vis->amplitude.data;
                        double2* s_ = (double2*)stokes.data;
                        s_[i].x = 0.5 * (d_[i].a.x - d_[i].d.x);
                        s_[i].y = 0.5 * (d_[i].a.y - d_[i].d.y);
                    }
                    else
                    {
                        float4c* d_ = (float4c*)vis->amplitude.data;
                        float2* s_ = (float2*)stokes.data;
                        s_[i].x = 0.5 * (d_[i].a.x - d_[i].d.x);
                        s_[i].y = 0.5 * (d_[i].a.y - d_[i].d.y);
                    }
                }
                break;
            }
            case OSKAR_IMAGE_TYPE_STOKES_U:
            {
                oskar_mem_init(&stokes, type | OSKAR_COMPLEX, location, num_vis_amps, OSKAR_TRUE);
                for (i = 0; i < num_vis_amps; ++i)
                {
                    if (type == OSKAR_DOUBLE)
                    {
                        double4c* d_ = (double4c*)vis->amplitude.data;
                        double2* s_ = (double2*)stokes.data;
                        s_[i].x = 0.5 * (d_[i].b.x + d_[i].c.x);
                        s_[i].y = 0.5 * (d_[i].b.y + d_[i].c.y);
                    }
                    else
                    {
                        float4c* d_ = (float4c*)vis->amplitude.data;
                        float2* s_ = (float2*)stokes.data;
                        s_[i].x = 0.5 * (d_[i].b.x + d_[i].c.x);
                        s_[i].y = 0.5 * (d_[i].b.y + d_[i].c.y);
                    }
                }
                break;
            }
            case OSKAR_IMAGE_TYPE_STOKES_V:
            {
                oskar_mem_init(&stokes, type | OSKAR_COMPLEX, location, num_vis_amps, OSKAR_TRUE);
                for (i = 0; i < num_vis_amps; ++i)
                {
                    if (type == OSKAR_DOUBLE)
                    {
                        double4c* d_ = (double4c*)vis->amplitude.data;
                        double2* s_ = (double2*)stokes.data;
                        s_[i].x =  0.5 * (d_[i].b.y - d_[i].c.y);
                        s_[i].y = -0.5 * (d_[i].b.x - d_[i].c.x);
                    }
                    else
                    {
                        float4c* d_ = (float4c*)vis->amplitude.data;
                        float2* s_ = (float2*)stokes.data;
                        s_[i].x =  0.5 * (d_[i].b.y - d_[i].c.y);
                        s_[i].y = -0.5 * (d_[i].b.x - d_[i].c.x);
                    }
                }
                break;
            }
            default:
            {
                return OSKAR_ERR_BAD_DATA_TYPE;
            }
        }; /* switch (pol_type) */
    }
    else if (num_vis_pols == 4 && pol_type == OSKAR_IMAGE_TYPE_STOKES)
    {
        printf("** evaluating stokes parameters\n");
        oskar_mem_init(&stokes, vis->amplitude.type, location, num_vis_amps, OSKAR_TRUE);
        for (i = 0; i < num_vis_amps; ++i)
        {
            if (type == OSKAR_DOUBLE)
            {
                double4c* d_ = (double4c*)vis->amplitude.data;
                double4c* s_ = (double4c*)stokes.data;
                /* I = 0.5 (XX + YY) */
                s_[i].a.x =  0.5 * (d_[i].a.x + d_[i].d.x);
                s_[i].a.y =  0.5 * (d_[i].a.y + d_[i].d.y);
                /* Q = 0.5 (XX - YY)*/
                s_[i].b.x =  0.5 * (d_[i].a.x - d_[i].d.x);
                s_[i].b.y =  0.5 * (d_[i].a.y - d_[i].d.y);
                /* U = 0.5 (XY + YX) */
                s_[i].c.x =  0.5 * (d_[i].b.x + d_[i].c.x);
                s_[i].c.y =  0.5 * (d_[i].b.y + d_[i].c.y);
                /* V = -0.5i (XY - YX) */
                s_[i].d.x =  0.5 * (d_[i].b.y - d_[i].c.y);
                s_[i].d.y = -0.5 * (d_[i].b.x - d_[i].c.x);
            }
            else
            {
                float4c* d_ = (float4c*)vis->amplitude.data;
                float4c* s_ = (float4c*)stokes.data;
                /* I */
                s_[i].a.x =  0.5 * (d_[i].a.x + d_[i].d.x);
                s_[i].a.y =  0.5 * (d_[i].a.y + d_[i].d.y);
                /* Q */
                s_[i].b.x =  0.5 * (d_[i].a.x - d_[i].d.x);
                s_[i].b.y =  0.5 * (d_[i].a.y - d_[i].d.y);
                /* U */
                s_[i].c.x =  0.5 * (d_[i].b.x + d_[i].c.x);
                s_[i].c.y =  0.5 * (d_[i].b.y + d_[i].c.y);
                /* V */
                s_[i].d.x =  0.5 * (d_[i].b.y - d_[i].c.y);
                s_[i].d.y = -0.5 * (d_[i].b.x - d_[i].c.x);
            }
        }
    }


    /* ___ Setup the image ___ **/
    oskar_image_resize(image, size, size, num_pols, num_times, num_chan);
    /* Set image meta-data */
    /* Note: not changing the dimension order here from that defined in
     * oskar_image_init() */
    oskar_mem_copy(&image->settings_path, &vis->settings_path);
    image->centre_ra_deg      = vis->phase_centre_ra_deg;
    image->centre_dec_deg     = vis->phase_centre_dec_deg;
    image->fov_ra_deg         = settings->fov_deg;
    image->fov_dec_deg        = settings->fov_deg;
    image->time_start_mjd_utc = vis->time_start_mjd_utc +
            (time_range[0] * vis->time_inc_seconds * SEC2DAYS);
    image->time_inc_sec       = vis->time_inc_seconds;
    image->freq_start_hz      = vis->freq_start_hz +
            (chan_range[0] * vis->channel_bandwidth_hz);
    image->image_type         = pol_type;
    /* TODO think a bit more about how best to set this for all scenarios
     * channel averaging ... ? */
    image->freq_inc_hz        = (settings->channel_snapshots) ?
            vis->freq_inc_hz : 0.0;
    /* Note: mean, variance etc as these can't be defined for cubes! */


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
    else
    {
        num_vis = vis->num_baselines * vis->num_channels * vis->num_times;
        oskar_mem_init(&uu,  type, location, num_vis, OSKAR_TRUE);
        oskar_mem_init(&vv,  type, location, num_vis, OSKAR_TRUE);
        oskar_mem_init(&amp, type | OSKAR_COMPLEX, location, num_vis, OSKAR_TRUE);
    }

    oskar_mem_init(&uu_ptr, type, location, vis->num_baselines, OSKAR_FALSE);
    oskar_mem_init(&vv_ptr, type, location, vis->num_baselines, OSKAR_FALSE);
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
        int ch_ = chan_range[0] + c;

        freq0 = image->freq_start_hz + (image->freq_inc_hz) * c + vis->channel_bandwidth_hz/2.0;
        lambda0 = C_0 / freq0;

        for (t = 0; t < num_times; ++t)
        {
            int coord_offset;
            int ti_ = time_range[0] + t;

            /* ___ Get baseline coordinates needed for imaging ___ */

            /* Snapshots in frequency and time */
            if (settings->time_snapshots && settings->channel_snapshots)
            {
                coord_offset = ti_ * vis->num_baselines;
                oskar_mem_get_pointer(&uu, &vis->uu_metres, coord_offset, num_vis);
                oskar_mem_get_pointer(&vv, &vis->vv_metres, coord_offset, num_vis);
            }

            /* Snapshots in time, frequency synthesis */
            else if (settings->time_snapshots && !settings->channel_snapshots)
            {
                coord_offset = ti_ * vis->num_baselines;
                oskar_mem_get_pointer(&uu_ptr, &vis->uu_metres, coord_offset, num_vis);
                oskar_mem_get_pointer(&vv_ptr, &vis->vv_metres, coord_offset, num_vis);

                for (j = 0; j < (chan_range[1] - chan_range[0]); ++j)
                {
                    lambda  = 0.0;
                    scaling = lambda0/lambda;
                    for (i = 0; i < vis->num_baselines; ++i)
                    {
                        int idx = j*vis->num_baselines + i;
                        if (type == OSKAR_DOUBLE)
                        {
                            ((double*)uu.data)[idx] = ((double*)uu_ptr.data)[i] * scaling;
                            ((double*)vv.data)[idx] = ((double*)vv_ptr.data)[i] * scaling;
                        }
                        else
                        {
                            ((float*)uu.data)[idx] = ((float*)uu_ptr.data)[i] * scaling;
                            ((float*)vv.data)[idx] = ((float*)vv_ptr.data)[i] * scaling;
                        }
                    }
                }
            }

            /* Frequency and time synthesis */
            else
            {
                for (k = 0; k < (time_range[1] - time_range[0]); ++k)
                {
                    coord_offset = (time_range[0] + k) * vis->num_baselines;
                    oskar_mem_get_pointer(&uu_ptr, &vis->uu_metres, coord_offset, num_vis);
                    oskar_mem_get_pointer(&vv_ptr, &vis->vv_metres, coord_offset, num_vis);

                    for (j = 0; j < (chan_range[1] - chan_range[0]); ++j)
                    {
                        double lambda  = 0.0;
                        double scaling = lambda0/lambda;
                        for (i = 0; i < vis->num_baselines; ++i)
                        {
                            int idx = j*vis->num_baselines + i;
                            if (type == OSKAR_DOUBLE)
                            {
                                ((double*)uu.data)[idx] = ((double*)uu_ptr.data)[i] * scaling;
                                ((double*)vv.data)[idx] = ((double*)vv_ptr.data)[i] * scaling;
                            }
                            else
                            {
                                ((float*)uu.data)[idx] = ((float*)uu_ptr.data)[i] * scaling;
                                ((float*)vv.data)[idx] = ((float*)vv_ptr.data)[i] * scaling;
                            }
                        }
                    }
                }
            }

            for (p = 0; p < num_pols; ++p)
            {
                /* ___ Get visibility amplitudes for imaging ___ */
                /* Snapshots in both frequency and time */
                if (settings->time_snapshots && settings->channel_snapshots)
                {
                    printf("-- (%i %i)\n", settings->time_snapshots, settings->channel_snapshots);
                    if (num_pols == 4)
                    {
                        if (pol_type == OSKAR_IMAGE_TYPE_STOKES)
                        {
                            if (type == OSKAR_DOUBLE)
                            {
                                double4c* s_ = (double4c*)stokes.data;
                                double2* a_  = (double2*)amp.data;
                                int offset = (ch_ * vis->num_times * vis->num_baselines) +
                                        ti_ * vis->num_baselines;
                                printf("-- stokes, double\n");
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
                                printf("-- stokes, single\n");
                                return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
                            }
                        } /* if (pol_type == STOKES) */
                        if (pol_type == OSKAR_IMAGE_TYPE_POL_LINEAR)
                        {
                            if (type == OSKAR_DOUBLE)
                            {
                                double4c* v_;
                                double2* a_;
                                int idx;
                                printf("-- linear, double\n");

                                a_ = (double2*)amp.data;
                                v_ = (double4c*)vis->amplitude.data;
                                idx = (ch_ * vis->num_times * vis->num_baselines) +
                                        (ti_ * vis->num_baselines);

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
                                printf("-- linear, single\n");
                                return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
                            }
                        } /* if (pol_type == LINEAR) */
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
                                int idx = (ch_ * vis->num_times * vis->num_baselines) +
                                        ti_ * vis->num_baselines;
                                printf("-- I|Q|U|V, double\n");
                                for (j = 0; j < vis->num_baselines; ++j)
                                {
                                    a_[j].x = v_[idx + j].x;
                                    a_[j].y = v_[idx + j].y;
                                }
                            }
                            else /* type == OSKAR_SINGLE */
                            {
                                printf("-- I|Q|U|V, single\n");
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
                                double2* v_ = (double2*)vis->amplitude.data;
                                double2* a_ = (double2*)amp.data;
                                int idx = (ch_ * vis->num_times * vis->num_baselines) +
                                        ti_ * vis->num_baselines;
                                printf("-- XX|XY|YX|YY, double\n");
                                for (j = 0; j < vis->num_baselines; ++j)
                                {
                                    a_[j].x = v_[idx + j].x;
                                    a_[j].y = v_[idx + j].y;
                                }
                            }
                            else /* type == OSKAR_SINGLE */
                            {
                                printf("-- XX|XY|YX|YY, single\n");
                                return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
                            }
                        }
                    }
                    else /* num_pols != (1 | 4) */
                    {
                        return OSKAR_ERR_DIMENSION_MISMATCH;
                    }
                }

                /* Snapshots in time, frequency synthesis */
                else if (settings->time_snapshots && !settings->channel_snapshots)
                {
                    printf("-- (%i %i)\n", settings->time_snapshots, settings->channel_snapshots);
                    return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
                }

                /* Frequency and time synthesis */
                else
                {
                    printf("-- (%i %i)\n", settings->time_snapshots, settings->channel_snapshots);
                    return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
                }

                /* ___ make the image ___ */
                if (settings->dft)
                {
                    /* NOTE the copy in dft needs sorting out */
                    slice_offset = c * num_times * num_pols * num_pixels +
                            t * num_pols * num_pixels + p * num_pixels;
                    oskar_mem_get_pointer(&im_slice, &image->data,
                            slice_offset, num_pixels);
                    err = oskar_make_image_dft(&im_slice, &uu, &vv, &amp,
                            &l, &m, freq0);
                    if (err) return err;
                }
                else
                {
                    /*err = fft()*/
                    printf("-- FFT!\n");
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
    oskar_mem_free(&uu_ptr);
    oskar_mem_free(&vv_ptr);
    oskar_mem_free(&im_slice);

    return OSKAR_SUCCESS;
}



#ifdef __cplusplus
}
#endif
