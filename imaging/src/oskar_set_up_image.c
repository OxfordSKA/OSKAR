/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <oskar_set_up_image.h>
#include <oskar_image.h>
#include <oskar_evaluate_image_ranges.h>

#define SEC2DAYS 1.15740740740740740740741e-5

#ifdef __cplusplus
extern "C" {
#endif

oskar_Image* oskar_set_up_image(const oskar_Vis* vis,
        const oskar_SettingsImage* settings, int* status)
{
    int im_chan_range[2], im_time_range[2];
    int vis_chan_range[2], vis_time_range[2];
    double freq_inc = 0.0;
    int im_num_chan, im_num_times, im_type, num_pols = 0;
    oskar_Image* im = 0;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Polarisation settings. */
    im_type = settings->image_type;
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
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }

    /* Set the channel range for the image cube [output range]. */
    oskar_evaluate_image_range(im_chan_range, settings->channel_snapshots,
            settings->channel_range, oskar_vis_num_channels(vis), status);
    im_num_chan = im_chan_range[1] - im_chan_range[0] + 1;

    /* Set the time range for the image cube [output range]. */
    oskar_evaluate_image_range(im_time_range, settings->time_snapshots,
            settings->time_range, oskar_vis_num_times(vis), status);
    im_num_times = im_time_range[1] - im_time_range[0] + 1;

    /* Time and channel range for data. */
    oskar_evaluate_image_data_range(vis_chan_range, settings->channel_range,
            oskar_vis_num_channels(vis), status);
    oskar_evaluate_image_data_range(vis_time_range, settings->time_range,
            oskar_vis_num_times(vis), status);

    /* Create the image and size it. */
    if (*status) return 0;
    im = oskar_image_create(oskar_mem_precision(oskar_vis_amplitude_const(vis)),
            OSKAR_CPU, status);
    oskar_image_resize(im, settings->size, settings->size,
            num_pols, im_num_times, im_num_chan, status);

    /* Set image meta-data. */
    /* __Note__ the dimension order used here is assumed unchanged from that
     * defined in oskar_image_init() */
    oskar_mem_copy(oskar_image_settings_path(im),
            oskar_vis_settings_path_const(vis), status);

    oskar_image_set_fov(im, settings->fov_deg, settings->fov_deg);

    if (settings->direction_type == OSKAR_IMAGE_DIRECTION_OBSERVATION)
    {
        oskar_image_set_centre(im, oskar_vis_phase_centre_ra_deg(vis),
                oskar_vis_phase_centre_dec_deg(vis));
    }
    else if (settings->direction_type == OSKAR_IMAGE_DIRECTION_RA_DEC)
    {
        oskar_image_set_centre(im, settings->ra_deg, settings->dec_deg);
    }
    else
    {
        *status = OSKAR_ERR_SETTINGS_IMAGE;
        oskar_image_free(im, status);
        return 0;
    }

    /* TODO for time synthesis the time inc should be 0...? need to
     * determine difference between inc and integration time. */
    oskar_image_set_time(im, oskar_vis_time_start_mjd_utc(vis) +
            (vis_time_range[0] * oskar_vis_time_inc_sec(vis) * SEC2DAYS),
            (settings->time_snapshots) ? oskar_vis_time_inc_sec(vis) : 0.0);

    /* TODO for channel synthesis the channel inc should be 0...? need to
     * determine difference between inc and channel bandwidth. */
    freq_inc = (settings->channel_snapshots) ? oskar_vis_freq_inc_hz(vis) : 0.0;

    if (settings->channel_snapshots)
    {
        oskar_image_set_freq(im, oskar_vis_freq_start_hz(vis) +
                vis_chan_range[0] * oskar_vis_freq_inc_hz(vis), freq_inc);
    }
    else
    {
        double chan0 = (vis_chan_range[1] - vis_chan_range[0]) / 2.0;
        oskar_image_set_freq(im, oskar_vis_freq_start_hz(vis) +
                chan0 * oskar_vis_freq_inc_hz(vis), freq_inc);
    }
    oskar_image_set_type(im, settings->image_type);

    /* NOTE: maybe these shouldn't be hard-coded?!? */
    oskar_image_set_coord_frame(im, OSKAR_IMAGE_COORD_FRAME_EQUATORIAL);
    oskar_image_set_grid_type(im, OSKAR_IMAGE_GRID_TYPE_RECTILINEAR);

    return im;
}

#ifdef __cplusplus
}
#endif
