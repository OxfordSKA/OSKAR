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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define SEC2DAYS 1.15740740740740740740741e-5

#ifdef __cplusplus
extern "C" {
#endif

int oskar_make_image(oskar_Image* image, const oskar_Visibilities* vis,
        const oskar_SettingsImage* settings)
{
    oskar_Mem l, m, stokes, uu, vv, amp, uu_ptr, vv_ptr;
    int t, c, p, i, j; /* loop indices */
    int type;
    int size, num_pixels, location, num_pols, num_times, num_chan; /* dims */
    int pol_type;
    int time_range[2], chan_range[2];
    int num_vis_amps, num_vis_pols;
    int num_vis; /* number of visibilities passed to image per plane of the cube */
    /*int coord_offset, amp_offset;*/
    double fov, freq;
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
    if (num_vis_pols > 1 && !(pol_type == OSKAR_IMAGE_TYPE_POL_LINEAR ||
            pol_type == OSKAR_IMAGE_TYPE_POL_XX || pol_type == OSKAR_IMAGE_TYPE_POL_YY ||
            pol_type == OSKAR_IMAGE_TYPE_POL_XY || pol_type == OSKAR_IMAGE_TYPE_POL_YX))
    {
        if (pol_type == OSKAR_IMAGE_TYPE_STOKES_I)
        {
            oskar_mem_init(&stokes, type, location, num_vis_amps, OSKAR_TRUE);
            /* TODO */
        }
        else if (pol_type == OSKAR_IMAGE_TYPE_STOKES_Q)
        {
            oskar_mem_init(&stokes, type, location, num_vis_amps, OSKAR_TRUE);
            /* TODO */
        }
        else if (pol_type == OSKAR_IMAGE_TYPE_STOKES_U)
        {
            oskar_mem_init(&stokes, type, location, num_vis_amps, OSKAR_TRUE);
            /* TODO */
        }
        else if (pol_type == OSKAR_IMAGE_TYPE_STOKES_V)
        {
            oskar_mem_init(&stokes, type, location, num_vis_amps, OSKAR_TRUE);
            /* TODO */
        }
        else if (pol_type == OSKAR_IMAGE_TYPE_STOKES)
        {
            oskar_mem_init(&stokes, type, location, num_vis_amps * 4, OSKAR_TRUE);
            /* TODO */
        }
    }

    /* ___ Setup the image ___ **/
    oskar_image_resize(image, size, size, num_pols, num_times, num_chan);
    /* Set image meta-data */
    /* Note: not changing the dimension order here from that defined in
     * oskar_image_init() */
    image->settings_path      = vis->settings_path;
    image->centre_ra_deg      = vis->phase_centre_ra_deg;
    image->centre_dec_deg     = vis->phase_centre_dec_deg;
    image->fov_ra_deg         = settings->fov_deg;
    image->fov_dec_deg        = settings->fov_deg;
    image->time_start_mjd_utc = vis->time_start_mjd_utc +
            (time_range[0] * vis->time_inc_seconds * SEC2DAYS);
    image->time_inc_sec       = vis->time_inc_seconds;
    image->freq_start_hz      = vis->freq_start_hz +
            (chan_range[0] * vis->channel_bandwidth_hz);
    image->freq_inc_hz        = vis->channel_bandwidth_hz;
    image->image_type         = pol_type;
    /* Note: mean, variance etc as these can't be defined for cubes! */


    /* Note: vis are channel -> time -> baseline order currently  */
    /*       vis coordinates are of length = num_times * num_baselines */
    /*       vis amp is of length = num_channels * num_times * num_baselines */
    if (settings->time_snapshots && settings->channel_snapshots)
    {
        num_vis = vis->num_baselines;
        oskar_mem_init(&uu,  type, location, num_vis, OSKAR_FALSE);
        oskar_mem_init(&vv,  type, location, num_vis, OSKAR_FALSE);
        oskar_mem_init(&amp, type, location, num_vis, OSKAR_FALSE);
    }
    else if (settings->time_snapshots && !settings->channel_snapshots)
    {
        num_vis = vis->num_baselines * vis->num_channels;
        oskar_mem_init(&uu,  type, location, num_vis, OSKAR_TRUE);
        oskar_mem_init(&vv,  type, location, num_vis, OSKAR_TRUE);
        oskar_mem_init(&amp, type, location, num_vis, OSKAR_TRUE);
    }
    else
    {
        num_vis = vis->num_baselines * vis->num_channels * vis->num_times;
        oskar_mem_init(&uu,  type, location, num_vis, OSKAR_TRUE);
        oskar_mem_init(&vv,  type, location, num_vis, OSKAR_TRUE);
        oskar_mem_init(&amp, type, location, num_vis, OSKAR_TRUE);
    }

    oskar_mem_init(&uu_ptr, type, location, vis->num_baselines, OSKAR_FALSE);
    oskar_mem_init(&vv_ptr, type, location, vis->num_baselines, OSKAR_FALSE);

    /* ___ Make the image ___ */
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

    for (c = 0; c < num_chan; ++c)
    {
        freq = 0.0;
        for (t = 0; t < num_times; ++t)
        {
            int coord_offset;
            coord_offset = time_range[0] + t * vis->num_baselines;
            oskar_mem_get_pointer(&uu_ptr, &vis->uu_metres, coord_offset, num_vis);
            oskar_mem_get_pointer(&vv_ptr, &vis->vv_metres, coord_offset, num_vis);

            /* ___ Get baseline coordinates needed for imaging ___ */
            /* Snapshots in frequency and time */
            if (settings->time_snapshots && settings->channel_snapshots)
            {
                oskar_mem_assign(&uu, &uu_ptr);
                oskar_mem_assign(&vv, &vv_ptr);
            }
            /* Snapshots in time, frequency synthesis */
            else if (settings->time_snapshots && !settings->channel_snapshots)
            {
                /* Grab the set of coordinates for the snapshot
                 * and duplicate them with frequency scaling
                 * -- scale by (lambda0 / lambda) where lambda0 is the
                 * centre frequency and that passed to the imager.*/
                for (j = 0; j < (chan_range[1] - chan_range[0]); ++j)
                {
                    for (i = 0; i < vis->num_baselines; ++i)
                    {
                        if (type == OSKAR_DOUBLE)
                        {
                            /*
                            double* uu_ = (double*)uu.data;
                            ((double*)uu.data)[j*vis->num_baselines + i] =
                                    ((double*)vv.data)[j*vis->num_baselines + i] =
                             */
                         }
                        else
                        {

                        }
                    }
                }


            }
            /* Frequency and time synthesis */
            else
            {
            }



            for (p = 0; p < num_pols; ++p)
            {
                /* Get visibility amplitudes for imaging */
                /* Snapshots in both frequency and time */
                if (settings->time_snapshots && settings->channel_snapshots)
                {
                    if (num_pols == 4)
                    {
                        if (pol_type == OSKAR_IMAGE_TYPE_STOKES)
                        {
                            /* TODO amp is set to an offset into Stokes */
                        }
                        if (pol_type == OSKAR_IMAGE_TYPE_POL_LINEAR)
                        {
                            /* TODO amp is set to an offset into vis->amp */
                        }
                    }
                    else
                    {
                        if (pol_type == OSKAR_IMAGE_TYPE_STOKES_I ||
                                pol_type == OSKAR_IMAGE_TYPE_STOKES_Q ||
                                pol_type == OSKAR_IMAGE_TYPE_STOKES_U ||
                                pol_type == OSKAR_IMAGE_TYPE_STOKES_V)
                        {
                            /* TODO Amp is set to offset into stokes */
                        }
                        else if (pol_type == OSKAR_IMAGE_TYPE_POL_XX ||
                                pol_type == OSKAR_IMAGE_TYPE_POL_XY ||
                                pol_type == OSKAR_IMAGE_TYPE_POL_XY ||
                                pol_type == OSKAR_IMAGE_TYPE_POL_YX)
                        {
                            /* TODO Amp is set to offset info vis->amp */
                        }
                    }
                }
                /* Snapshots in time, frequency synthesis */
                else if (settings->time_snapshots && !settings->channel_snapshots)
                {
                }
                /* Frequency and time synthesis */
                else
                {
                }


                if (settings->dft)
                {
                    /* NOTE the copy in dft needs sorting out */
                    err = oskar_make_image_dft(&image->data, &uu, &vv, &amp,
                            &l, &m, freq);
                    if (err) return err;
                }
                else
                {
                    /*err = fft()*/
                    return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
                }
            }
        }
    }


    /* Clean up */
    oskar_mem_free(&l);
    oskar_mem_free(&m);
    oskar_mem_free(&stokes);
    oskar_mem_free(&uu);
    oskar_mem_free(&vv);
    oskar_mem_free(&amp);
    oskar_mem_free(&uu_ptr);
    oskar_mem_free(&vv_ptr);

    return OSKAR_SUCCESS;
}



#ifdef __cplusplus
}
#endif
