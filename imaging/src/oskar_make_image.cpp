/*
 * Copyright (c) 2012-2013, The University of Oxford
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
#include "imaging/oskar_image_evaluate_ranges.h"
#include "oskar_convert_apparent_ra_dec_to_relative_direction_cosines.h"
#include "oskar_convert_ecef_to_baseline_uvw.h"

#include <oskar_log.h>
#include <oskar_mem.h>
#include <oskar_get_data_type_string.h>

#include <math.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEG2RAD M_PI/180.0

#define SEC2DAYS 1.15740740740740740740741e-5

/* uu, vv, ww are in metres */
static void phase_rotate_vis_amps(oskar_Mem* amps, int num_vis, int type,
        double delta_l, double delta_m, double delta_n, const oskar_Mem* uu,
        const oskar_Mem* vv, const oskar_Mem* ww, double freq);

#ifdef __cplusplus
extern "C" {
#endif

int oskar_make_image(oskar_Image* im, oskar_Log* log,
        const oskar_Vis* vis, const oskar_SettingsImage* settings)
{
    int err = OSKAR_SUCCESS;

    /* Location of temporary memory used by this function (needs to be CPU). */
    int location = OSKAR_LOCATION_CPU;

    if (im == NULL || vis == NULL || settings == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Local variables. */
    int num_channels = oskar_vis_num_channels(vis);
    int num_times = oskar_vis_num_times(vis);
    int num_stations = oskar_vis_num_stations(vis);
    int num_baselines = oskar_vis_num_baselines(vis);
    double freq_start_hz = oskar_vis_freq_start_hz(vis);
    double freq_inc_hz = oskar_vis_freq_inc_hz(vis);
    const oskar_Mem *baseline_uu = oskar_vis_baseline_uu_metres_const(vis);
    const oskar_Mem *baseline_vv = oskar_vis_baseline_vv_metres_const(vis);
    const oskar_Mem *baseline_ww = oskar_vis_baseline_ww_metres_const(vis);
    const oskar_Mem *station_ecef_x = oskar_vis_station_x_metres_const(vis);
    const oskar_Mem *station_ecef_y = oskar_vis_station_y_metres_const(vis);
    const oskar_Mem *station_ecef_z = oskar_vis_station_z_metres_const(vis);

    int type = (oskar_mem_is_double(oskar_vis_amplitude_const(vis)) &&
            oskar_mem_is_double(&im->data)) ? OSKAR_DOUBLE : OSKAR_SINGLE;

    /* If imaging away from the beam direction, evaluate l0-l, m0-m, n0-n
     * for the new pointing centre as well as a set of baseline coordinates
     * corresponding to the user specified imaging direction.
     */
    double delta_l = 0.0, delta_m = 0.0, delta_n = 0.0;
    oskar_Mem uu_rot, vv_rot, ww_rot;
    if (settings->direction_type == OSKAR_IMAGE_DIRECTION_RA_DEC)
    {
        double ra_rad = settings->ra_deg * DEG2RAD;
        double dec_rad = settings->dec_deg * DEG2RAD;
        double ra0_rad = oskar_vis_phase_centre_ra_deg(vis) * DEG2RAD;
        double dec0_rad = oskar_vis_phase_centre_dec_deg(vis) * DEG2RAD;
        double l1, m1, n1;
        int num_elements = num_baselines * num_times;

        oskar_convert_apparent_ra_dec_to_relative_direction_cosines_d(1,
                &ra_rad, &dec_rad, ra0_rad, dec0_rad, &l1, &m1, &n1);
        delta_l = 0 - l1;
        delta_m = 0 - m1;
        delta_n = 1 - n1;

        oskar_mem_init(&uu_rot, type, location, num_elements, OSKAR_TRUE, &err);
        oskar_mem_init(&vv_rot, type, location, num_elements, OSKAR_TRUE, &err);
        oskar_mem_init(&ww_rot, type, location, num_elements, OSKAR_TRUE, &err);

        /* Work array for baseline evaluation. */
        oskar_Mem work_uvw;
        oskar_mem_init(&work_uvw, type, location, 3 * num_stations, 1, &err);
        oskar_convert_ecef_to_baseline_uvw(&uu_rot, &vv_rot, &ww_rot,
                num_stations, station_ecef_x, station_ecef_y, station_ecef_z,
                ra_rad, dec_rad, num_times, oskar_vis_time_start_mjd_utc(vis),
                oskar_vis_time_inc_seconds(vis) * SEC2DAYS, &work_uvw, &err);
        oskar_mem_free(&work_uvw, &err);
    }

    /* Initialise the OSKAR image cube */
    oskar_image_init(im, type, location, &err);
    if (err) return err;

    int size = settings->size;
    double fov = settings->fov_deg * M_PI/180.0;

    /* Set the channel range for the image cube [output range]. */
    int im_chan_range[2];
    err = oskar_evaluate_image_range(im_chan_range, settings->channel_snapshots,
            settings->channel_range, num_channels);
    if (err) return err;
    int im_num_chan = im_chan_range[1] - im_chan_range[0] + 1;

    /* Set the time range for the image cube [output range]. */
    int im_time_range[2];
    err = oskar_evaluate_image_range(im_time_range, settings->time_snapshots,
            settings->time_range, num_times);
    if (err) return err;
    int im_num_times = im_time_range[1] - im_time_range[0] + 1;

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

    int num_vis_pols = oskar_vis_num_polarisations(vis);
    if (im_num_times > num_times || im_num_chan > num_channels ||
            num_pols > num_vis_pols)
    {
        return OSKAR_ERR_DIMENSION_MISMATCH;
    }
    if (num_vis_pols == 1 && !(im_type == OSKAR_IMAGE_TYPE_STOKES_I ||
            im_type == OSKAR_IMAGE_TYPE_PSF))
    {
        return OSKAR_ERR_SETTINGS;
    }

    /* Evaluate Stokes parameters  (if required) */
    oskar_Mem stokes;
    oskar_mem_init(&stokes, type, location, 0, OSKAR_FALSE, &err);
    err = oskar_get_image_stokes(&stokes, vis, settings);
    if (err) return err;

    int num_vis = 0;
    if (settings->time_snapshots && settings->channel_snapshots)
        num_vis = num_baselines;
    else if (settings->time_snapshots && !settings->channel_snapshots)
        num_vis = num_baselines * num_channels;
    else if (!settings->time_snapshots && settings->channel_snapshots)
        num_vis = num_baselines * num_times;
    else /* Time and frequency synthesis */
        num_vis = num_baselines * num_channels * num_times;
    oskar_Mem uu_im, vv_im, ww_im, vis_im, uu_tmp, vv_tmp, ww_tmp, l, m;
    oskar_mem_init(&uu_im, type, location, num_vis, 1, &err);
    oskar_mem_init(&vv_im, type, location, num_vis, 1, &err);
    oskar_mem_init(&ww_im, type, location, num_vis, 1, &err);
    oskar_mem_init(&vis_im, type | OSKAR_COMPLEX, location, num_vis, 1, &err);
    if (settings->direction_type == OSKAR_IMAGE_DIRECTION_RA_DEC)
    {
        oskar_mem_init(&uu_tmp, type, location, num_vis, OSKAR_TRUE, &err);
        oskar_mem_init(&vv_tmp, type, location, num_vis, OSKAR_TRUE, &err);
        oskar_mem_init(&ww_tmp, type, location, num_vis, OSKAR_TRUE, &err);
    }
    if (err) return err;

    /* Allocate pixel coordinate grid required for the DFT imager. */
    oskar_mem_init(&l, type, location, num_pixels, 1, &err);
    oskar_mem_init(&m, type, location, num_pixels, 1, &err);
    if (type == OSKAR_SINGLE)
    {
        oskar_evaluate_image_lm_grid_f(size, size, fov, fov,
                oskar_mem_float(&l, &err), oskar_mem_float(&m, &err));
    }
    else
    {
        oskar_evaluate_image_lm_grid_d(size, size, fov, fov,
                oskar_mem_double(&l, &err), oskar_mem_double(&m, &err));
    }

    /* Set up the image cube. */
    err = oskar_setup_image(im, vis, settings);
    if (err) return err;

    /* Declare a pointer to the slice of the image cube being imaged. */
    oskar_Mem im_slice;
    oskar_mem_init(&im_slice, type, location, num_pixels, OSKAR_FALSE, &err);

    /* Construct the image cube. */
    for (int i = 0, c = 0; c < im_num_chan; ++c)
    {
        int vis_chan = im_chan_range[0] + c;
        double im_freq = im->freq_start_hz + c * im->freq_inc_hz;
        oskar_log_message(log, 0, "Channel %3d/%d [%.4f MHz]",
                c + 1, im_num_chan, im_freq / 1e6);

        for (int t = 0; t < im_num_times; ++t)
        {
            int vis_time = im_time_range[0] + t;

            /* Evaluate baseline coordinates needed for imaging. */
            if (settings->direction_type == OSKAR_IMAGE_DIRECTION_OBSERVATION)
            {
                err = oskar_get_image_baseline_coords(&uu_im, &vv_im, &ww_im,
                        baseline_uu, baseline_vv, baseline_ww, num_times,
                        num_baselines, num_channels, freq_start_hz,
                        freq_inc_hz, vis_time, im_freq, settings);
                if (err) return err;
            }
            else if (settings->direction_type == OSKAR_IMAGE_DIRECTION_RA_DEC)
            {
                /* Rotated coordinates (used for imaging) */
                err = oskar_get_image_baseline_coords(&uu_im, &vv_im, &ww_im,
                        &uu_rot, &vv_rot, &ww_rot, num_times,
                        num_baselines, num_channels, freq_start_hz,
                        freq_inc_hz, vis_time, im_freq, settings);
                if (err) return err;

                /* Unrotated coordinates (used for phase rotation) */
                err = oskar_get_image_baseline_coords(&uu_tmp, &vv_tmp, &ww_tmp,
                        baseline_uu, baseline_vv, baseline_ww, num_times,
                        num_baselines, num_channels, freq_start_hz,
                        freq_inc_hz, vis_time, im_freq, settings);
                if (err) return err;
            }
            else
                return OSKAR_ERR_SETTINGS_IMAGE;

            for (int p = 0; p < num_pols; ++p, ++i)
            {
                oskar_log_message(log, 1, "Making image %3i/%i, "
                        "cube index (c=%i, t=%i, p=%i)",
                        i+1, (im_num_chan*im_num_times*num_pols), c, t, p);

                /* Get visibility amplitudes for imaging. */
                if (im_type == OSKAR_IMAGE_TYPE_PSF)
                {
                    oskar_mem_set_value_real(&vis_im, 1.0, 0, 0, &err);
                    if (err) return err;
                }
                else
                {
                    err = oskar_get_image_vis_amps(&vis_im, vis, &stokes, settings,
                            vis_chan, vis_time, p);
                    if (err) return err;

                    /* Phase rotate the visibilities. */
                    if (settings->direction_type == OSKAR_IMAGE_DIRECTION_RA_DEC)
                    {
                        phase_rotate_vis_amps(&vis_im, num_vis, type,
                                delta_l, delta_m, delta_n, &uu_tmp, &vv_tmp,
                                &ww_tmp, im_freq);
                    }
                }

                oskar_log_message(log, 2, "Number of visibilities %i", (int)oskar_mem_length(&vis_im));

                /* Get pointer to slice of the image cube. */
                int slice_offset = ((c * im_num_times + t) * num_pols + p) * num_pixels;
                oskar_mem_get_pointer(&im_slice, &im->data, slice_offset,
                        num_pixels, &err);
                if (err) return err;

                /* Make the image */
                if (settings->transform_type == OSKAR_IMAGE_DFT_2D)
                {
                    err = oskar_make_image_dft(&im_slice, &uu_im, &vv_im, &vis_im,
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

    oskar_mem_free(&uu_rot, &err);
    oskar_mem_free(&vv_rot, &err);
    oskar_mem_free(&ww_rot, &err);
    oskar_mem_free(&stokes, &err);
    oskar_mem_free(&uu_tmp, &err);
    oskar_mem_free(&vv_tmp, &err);
    oskar_mem_free(&ww_tmp, &err);
    oskar_mem_free(&uu_im, &err);
    oskar_mem_free(&vv_im, &err);
    oskar_mem_free(&ww_im, &err);
    oskar_mem_free(&vis_im, &err);
    oskar_mem_free(&l, &err);
    oskar_mem_free(&m, &err);
    oskar_mem_free(&im_slice, &err);
    return err;
}


#ifdef __cplusplus
}
#endif


/*
 * TODO Make this a stand-alone function?
 *
 * Ref:
 * Cornwell, T.J., & Perley, R.A., 1992,
 * "Radio-interferometric imaging of very large fields"
 */
static void phase_rotate_vis_amps(oskar_Mem* amps, int num_vis, int type,
        double delta_l, double delta_m, double delta_n, const oskar_Mem* uu,
        const oskar_Mem* vv, const oskar_Mem* ww, double freq)
{
    int i;
    double inv_lambda = freq / 299792458.0;

    /* This would be also easier as can avoid polarisation this way too... */
    if (type == OSKAR_DOUBLE)
    {
        double* uu_ = (double*)uu->data;
        double* vv_ = (double*)vv->data;
        double* ww_ = (double*)ww->data;
        double2* amp_ = (double2*)amps->data;

        for (i = 0; i < num_vis; ++i)
        {
            double u = uu_[i] * inv_lambda;
            double v = vv_[i] * inv_lambda;
            double w = ww_[i] * inv_lambda;
            double arg = 2.0 * M_PI * (u * delta_l + v * delta_m + w * delta_n);
            double phase_re = cos(arg);
            double phase_im = sin(arg);
            double re = amp_[i].x * phase_re - amp_[i].y * phase_im;
            double im = amp_[i].x * phase_im + amp_[i].y * phase_re;
            amp_[i].x = re;
            amp_[i].y = im;
        }
    }
    else
    {
        float* uu_ = (float*)uu->data;
        float* vv_ = (float*)vv->data;
        float* ww_ = (float*)ww->data;
        float2* amp_ = (float2*)amps->data;

        for (i = 0; i < num_vis; ++i)
        {
            float u = uu_[i] * inv_lambda;
            float v = vv_[i] * inv_lambda;
            float w = ww_[i] * inv_lambda;
            float arg = 2.0 * M_PI * (u * delta_l + v * delta_m + w * delta_n);
            float phase_re = cosf(arg);
            float phase_im = sinf(arg);
            float re = amp_[i].x * phase_re - amp_[i].y * phase_im;
            float im = amp_[i].x * phase_im + amp_[i].y * phase_re;
            amp_[i].x = re;
            amp_[i].y = im;
        }
    }
}
