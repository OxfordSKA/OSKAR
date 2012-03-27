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
#include "imaging/oskar_get_image_vis_amps.h"
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
    int t, c, p; /* loop indices */
    int type;
    int size, num_pixels, location, num_pols, num_times, num_chan; /* dims */
    int pol_type;
    int time_range[2], chan_range[2];
    int num_vis_pols;
    int num_vis; /* number of visibilities passed to image per plane of the cube */
    double fov, freq0;
    int slice_offset;
    int err = OSKAR_SUCCESS;

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
    if (time_range[0] < 0) time_range[0] = 0;
    if (time_range[1] < 0) time_range[1] = (settings->time_snapshots) ?
            vis->num_times-1 : 0;
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
    err = oskar_mem_init(&stokes, type, location, 0, OSKAR_FALSE);
    if (err) return err;
    err = oskar_get_image_stokes(&stokes, vis, settings);
    if (err) return err;

    /* Setup the image */
    err = oskar_setup_image(im, vis, settings);
    if (err) return err;

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

//    printf("INFO: channel snapshots = %s\n",
//            settings->channel_snapshots ? "true":"false");
//    printf("INFO: channel range  = %i %i\n", chan_range[0], chan_range[1]);
//    printf("INFO: num channels   = %i\n", num_chan);
//    printf("INFO: time snapshots = %s\n",
//            settings->time_snapshots ? "true":"false");
//    printf("INFO: time range     = %i %i\n", time_range[0], time_range[1]);
//    printf("INFO: num times      = %i\n", num_times);
//    printf("INFO: num pols       = %i\n", num_pols);

    /* ___ Make the image ___ */
//    int slice = 0;
    for (c = 0; c < num_chan; ++c)
    {
        int channel = chan_range[0] + c;
        freq0 = im->freq_start_hz + (im->freq_inc_hz) * c + vis->channel_bandwidth_hz/2.0;

        for (t = 0; t < num_times; ++t)
        {
            int time = time_range[0] + t;

            /* Evaluate baseline coords needed for imaging */
            err = oskar_get_image_baseline_coords(&uu, &vv, vis, time, settings);
            if (err) return err;

            for (p = 0; p < num_pols; ++p)
            {
                /* ___ Get visibility amplitudes for imaging ___ */
                err = oskar_get_image_vis_amps(&amp, vis, &stokes, settings,
                        channel, time, p);
                if (err) return err;

                /* Get a pointer to the slice of the image currently being
                 * imaged */
                slice_offset = ((c * num_times + t) * num_pols + p) * num_pixels;
                err = oskar_mem_get_pointer(&im_slice, &im->data,
                        slice_offset, num_pixels);
                if (err) return err;

                /* ___ make the image ___ */
                if (settings->dft)
                {
//                    char sliceid[20];
//                    sprintf(sliceid, "c%i_t%i_p%i", c, t, p);
//                    uu.binary_file_write_ext("im.dat",  "uu",  sliceid, slice);
//                    vv.binary_file_write_ext("im.dat",  "vv",  sliceid, slice);
//                    amp.binary_file_write_ext("im.dat", "amp", sliceid, slice);
//                    slice++;
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

    return err;
}

#ifdef __cplusplus
}
#endif
