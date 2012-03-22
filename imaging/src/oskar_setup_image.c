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


#include "imaging/oskar_setup_image.h"
#include "imaging/oskar_image_resize.h"
#include "utility/oskar_mem_copy.h"

#define SEC2DAYS 1.15740740740740740740741e-5

#ifdef __cplusplus
extern "C" {
#endif

int oskar_setup_image(oskar_Image* im, const oskar_Visibilities* vis,
        const oskar_SettingsImage* settings)
{
    int pol;
    int num_pols;
    int num_times;
    int num_chan;

    /* Set local variables */
    pol = settings->polarisation;
    if (pol == OSKAR_IMAGE_TYPE_STOKES_I ||
            pol == OSKAR_IMAGE_TYPE_STOKES_Q ||
            pol == OSKAR_IMAGE_TYPE_STOKES_U ||
            pol == OSKAR_IMAGE_TYPE_STOKES_V ||
            pol == OSKAR_IMAGE_TYPE_POL_XX ||
            pol == OSKAR_IMAGE_TYPE_POL_YY ||
            pol == OSKAR_IMAGE_TYPE_POL_XY ||
            pol == OSKAR_IMAGE_TYPE_POL_YX)
    {
        num_pols = 1;
    }
    else if (pol == OSKAR_IMAGE_TYPE_STOKES ||
            pol == OSKAR_IMAGE_TYPE_POL_LINEAR)
    {
        num_pols = 4;
    }
    else
    {
        return OSKAR_ERR_BAD_DATA_TYPE;
    }
    num_times = (settings->time_snapshots) ?
            (settings->time_range[1] - settings->time_range[0] + 1) : 1;
    if (num_times < 1) return OSKAR_ERR_INVALID_RANGE;
    num_chan  = (settings->channel_snapshots) ?
            (settings->channel_range[1] - settings->channel_range[0] + 1) : 1;
    if (num_chan < 1) return OSKAR_ERR_INVALID_RANGE;

    /* Resize the image cube */
    oskar_image_resize(im, settings->size, settings->size,
            num_pols, num_times, num_chan);

    /* Set image meta-data */
    /* Note: not changing the dimension order here from that defined in
     * oskar_image_init() */
    oskar_mem_copy(&im->settings_path, &vis->settings_path);
    im->centre_ra_deg      = vis->phase_centre_ra_deg;
    im->centre_dec_deg     = vis->phase_centre_dec_deg;
    im->fov_ra_deg         = settings->fov_deg;
    im->fov_dec_deg        = settings->fov_deg;
    im->time_start_mjd_utc = vis->time_start_mjd_utc +
            (settings->time_range[0] * vis->time_inc_seconds * SEC2DAYS);
    im->time_inc_sec       = vis->time_inc_seconds;
    if (settings->channel_snapshots)
    {
        im->freq_start_hz = vis->freq_start_hz +
                (settings->channel_range[0] * vis->freq_inc_hz);
        im->freq_inc_hz = vis->freq_inc_hz;
    }
    else
    {
        im->freq_start_hz = vis->freq_start_hz + (settings->channel_range[0] * vis->freq_inc_hz) +
                ((settings->channel_range[1]-settings->channel_range[0]) *  vis->freq_inc_hz) / 2.0;
        im->freq_inc_hz = 0.0;
    }
    im->freq_inc_hz        = (settings->channel_snapshots) ? vis->freq_inc_hz : 0.0;
    im->image_type         = settings->polarisation;
    /* Note: mean, variance etc as these can't be defined for cubes! */

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
