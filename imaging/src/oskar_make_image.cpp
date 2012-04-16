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
#include "imaging/oskar_image_init.h"

#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_mem_get_pointer.h"
#include "utility/oskar_mem_copy.h"
#include "utility/oskar_mem_assign.h"
#include "utility/oskar_mem_copy.h"
#include "utility/oskar_vector_types.h"
#include "utility/oskar_get_data_type_string.h"

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
    oskar_Mem l, m, stokes, temp_uu, temp_vv, temp_amp, im_slice;
    int err = OSKAR_SUCCESS;

    // The location of temporary memory used by this function (needs to be CPU).
    int location = OSKAR_LOCATION_CPU;

    // Local variables.
    if (im == NULL || vis == NULL || settings == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;
    int type = (oskar_mem_is_double(vis->amplitude.type) &&
            oskar_mem_is_double(im->data.type)) ? OSKAR_DOUBLE : OSKAR_SINGLE;

    err = oskar_image_init(im, type, location);
    if (err) return err;

    int size = settings->size;
    double fov = settings->fov_deg * M_PI/180.0;

    // Set the channel range for the image cube [output range].
    int im_chan_range[2];
    im_chan_range[0] = (settings->channel_range[0] < 0) ? 0 : settings->channel_range[0];
    if (settings->channel_range[1] < 0)
        im_chan_range[1] = settings->channel_snapshots ? vis->num_channels-1 : 0;
    else
        im_chan_range[1] = settings->channel_range[1];
    if (!settings->channel_snapshots && im_chan_range[1] > 0)
        return OSKAR_ERR_INVALID_RANGE;
    int num_chan  = (settings->channel_snapshots) ?
            (im_chan_range[1] - im_chan_range[0] + 1) : 1;
    if (num_chan < 1) return OSKAR_ERR_INVALID_RANGE;

    // Set the time range for the image cube [output range].
    int im_time_range[2];
    im_time_range[0] = (settings->time_range[0] < 0) ? 0 : settings->time_range[0];
    if (settings->time_range[1] < 0)
        im_time_range[1] = settings->time_snapshots ? vis->num_times-1 : 0;
    else
        im_time_range[1] = settings->time_range[1];
    if (!settings->time_snapshots && im_time_range[1] > 0)
        return OSKAR_ERR_INVALID_RANGE;
    int num_times = (settings->time_snapshots) ?
            (im_time_range[1] - im_time_range[0] + 1) : 1;
    if (num_times < 1) return OSKAR_ERR_INVALID_RANGE;

    int num_pixels = size*size;
    int im_type = settings->image_type;
    int num_pols = 0;
    if (im_type == OSKAR_IMAGE_TYPE_STOKES_I ||
            im_type == OSKAR_IMAGE_TYPE_STOKES_Q ||
            im_type == OSKAR_IMAGE_TYPE_STOKES_U ||
            im_type == OSKAR_IMAGE_TYPE_STOKES_V ||
            im_type == OSKAR_IMAGE_TYPE_POL_XX ||
            im_type == OSKAR_IMAGE_TYPE_POL_YY ||
            im_type == OSKAR_IMAGE_TYPE_POL_XY ||
            im_type == OSKAR_IMAGE_TYPE_POL_YX ||
            im_type == OSKAR_IMAGE_TYPE_PSF)
    {
        num_pols = 1;
    }
    else if (im_type == OSKAR_IMAGE_TYPE_STOKES ||
            im_type == OSKAR_IMAGE_TYPE_POL_LINEAR)
    {
        num_pols = 4;
    }
    else return OSKAR_ERR_BAD_DATA_TYPE;

    int num_vis_pols = oskar_mem_is_matrix(vis->amplitude.type) ? 4 : 1;
    if (num_times > vis->num_times || num_chan > vis->num_channels ||
            num_pols > num_vis_pols)
    {
        return OSKAR_ERR_DIMENSION_MISMATCH;
    }
    if (num_vis_pols == 1 && !(im_type == OSKAR_IMAGE_TYPE_STOKES_I ||
            im_type == OSKAR_IMAGE_TYPE_PSF))
        return OSKAR_ERR_SETTINGS;

    // Evaluate Stokes parameters  (if required)
    err = oskar_mem_init(&stokes, type, location, 0, OSKAR_FALSE);
    if (err) return err;
    err = oskar_get_image_stokes(&stokes, vis, settings);
    if (err) return err;

    /* Note: vis are channel -> time -> baseline order currently  */
    /*       vis coordinates are of length = num_times * num_baselines */
    /*       vis amp is of length = num_channels * num_times * num_baselines */
    int num_vis = 0;
    if (settings->time_snapshots && settings->channel_snapshots)
    {
        num_vis = vis->num_baselines;
        oskar_mem_init(&temp_uu,  type, location, num_vis, OSKAR_FALSE);
        oskar_mem_init(&temp_vv,  type, location, num_vis, OSKAR_FALSE);
        oskar_mem_init(&temp_amp, type | OSKAR_COMPLEX, location, num_vis, OSKAR_TRUE);
    }
    else if (settings->time_snapshots && !settings->channel_snapshots)
    {
        num_vis = vis->num_baselines * vis->num_channels;
        oskar_mem_init(&temp_uu,  type, location, num_vis, OSKAR_TRUE);
        oskar_mem_init(&temp_vv,  type, location, num_vis, OSKAR_TRUE);
        oskar_mem_init(&temp_amp, type | OSKAR_COMPLEX, location, num_vis, OSKAR_TRUE);
    }
    else if (!settings->time_snapshots && settings->channel_snapshots)
    {
        num_vis = vis->num_baselines * vis->num_times;
        oskar_mem_init(&temp_uu,  type, location, num_vis, OSKAR_TRUE);
        oskar_mem_init(&temp_vv,  type, location, num_vis, OSKAR_TRUE);
        oskar_mem_init(&temp_amp, type | OSKAR_COMPLEX, location, num_vis, OSKAR_TRUE);
    }
    else /* Time and frequency synthesis */
    {
        num_vis = vis->num_baselines * vis->num_channels * vis->num_times;
        oskar_mem_init(&temp_uu,  type, location, num_vis, OSKAR_TRUE);
        oskar_mem_init(&temp_vv,  type, location, num_vis, OSKAR_TRUE);
        oskar_mem_init(&temp_amp, type | OSKAR_COMPLEX, location, num_vis, OSKAR_TRUE);
    }

    // Allocate pixel coordinate grid required for the DFT imager.
    if (settings->transform_type == OSKAR_IMAGE_DFT_2D)
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

    // Setup the image cube.
    err = oskar_setup_image(im, vis, settings);
    if (err) return err;

    // Declare a pointer to the slice of the image cube being imaged.
    oskar_mem_init(&im_slice, type, location, num_pixels, OSKAR_FALSE);

    // Construct the image cube.
    for (int i = 0, c = 0; c < num_chan; ++c)
    {
        int vis_chan = im_chan_range[0] + c;
        double im_freq = im->freq_start_hz + c * im->freq_inc_hz;

        for (int t = 0; t < num_times; ++t)
        {
            int vis_time = im_time_range[0] + t;

            // Evaluate baseline coordinates needed for imaging.
            err = oskar_get_image_baseline_coords(&temp_uu, &temp_vv, vis, vis_time,
                    im_freq, settings);
            if (err) return err;

            for (int p = 0; p < num_pols; ++p, ++i)
            {
                printf("--> Making image [%3i/%3i]: freq = % -8.2f MHz, "
                        "cube index = (%i, %i, %i)\n",
                        i, (num_chan*num_times*num_pols), im_freq/1.0e6, c, t, p);

                // Get visibility amplitudes for imaging.
                if (im_type == OSKAR_IMAGE_TYPE_PSF)
                {
                    temp_amp.set_value_real(1.0);
                }
                else
                {
                    err = oskar_get_image_vis_amps(&temp_amp, vis, &stokes, settings,
                            vis_chan, vis_time, p);
                    if (err) return err;
                }

                // Get pointer to slice of the image cube.
                int slice_offset = ((c * num_times + t) * num_pols + p) * num_pixels;
                err = oskar_mem_get_pointer(&im_slice, &im->data, slice_offset,
                        num_pixels);
                if (err) return err;

                // Make the image
                if (settings->transform_type == OSKAR_IMAGE_DFT_2D)
                {
//                    temp_uu.binary_file_write_ext("im_data.dat", "data", "uu", i);
//                    temp_vv.binary_file_write_ext("im_data.dat", "data", "vv", i);
                    err = oskar_make_image_dft(&im_slice, &temp_uu, &temp_vv, &temp_amp,
                            &l, &m, im_freq);
                    if (err) return err;
                }
                else
                {
                    return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
                }
            }
        }
    }

    return err;
}

#ifdef __cplusplus
}
#endif
