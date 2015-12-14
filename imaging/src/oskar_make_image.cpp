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

#include <oskar_make_image.h>

#include <oskar_make_image_dft.h>
#include <oskar_evaluate_image_lm_grid.h>
#include <oskar_evaluate_image_ranges.h>
#include <oskar_get_image_baseline_coords.h>
#include <oskar_get_image_stokes.h>
#include <oskar_get_image_vis_amps.h>
#include <oskar_set_up_image.h>
#include <oskar_convert_ecef_to_baseline_uvw.h>
#include <oskar_convert_lon_lat_to_relative_directions.h>
#include <oskar_cmath.h>

#include <stddef.h>

#define DEG2RAD M_PI/180.0

#define SEC2DAYS 1.15740740740740740740741e-5

/* uu, vv, ww are in metres */
static void phase_rotate_vis_amps(oskar_Mem* amps, int num_vis, int type,
        double delta_l, double delta_m, double delta_n, const oskar_Mem* uu,
        const oskar_Mem* vv, const oskar_Mem* ww, double freq);

#ifdef __cplusplus
extern "C" {
#endif

oskar_Image* oskar_make_image(oskar_Log* log, const oskar_Vis* vis,
        const oskar_SettingsImage* settings, int* status)
{
    oskar_Image* im = 0;

    /* Location of temporary memory used by this function (needs to be CPU). */
    int location = OSKAR_CPU;

    /* Check if safe to proceed. */
    if (*status) return 0;

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
    const oskar_Mem *station_ecef_x =
            oskar_vis_station_x_offset_ecef_metres_const(vis);
    const oskar_Mem *station_ecef_y =
            oskar_vis_station_y_offset_ecef_metres_const(vis);
    const oskar_Mem *station_ecef_z =
            oskar_vis_station_z_offset_ecef_metres_const(vis);

    /* Get and check settings. */
    int size = settings->size;
    int num_pixels = size * size;
    double fov_rad = settings->fov_deg * M_PI/180.0;

    /* Set the channel range for the image cube [output range]. */
    int im_chan_range[2];
    oskar_evaluate_image_range(im_chan_range, settings->channel_snapshots,
            settings->channel_range, num_channels, status);
    int im_num_chan = im_chan_range[1] - im_chan_range[0] + 1;

    /* Set the time range for the image cube [output range]. */
    int im_time_range[2];
    oskar_evaluate_image_range(im_time_range, settings->time_snapshots,
            settings->time_range, num_times, status);
    if (*status) return 0;
    int im_num_times = im_time_range[1] - im_time_range[0] + 1;

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
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }

    /* Get number of polarisations. */
    int num_vis_pols = oskar_vis_num_pols(vis);
    if (im_num_times > num_times || im_num_chan > num_channels ||
            num_pols > num_vis_pols)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return 0;
    }
    if (num_vis_pols == 1 && !(im_type == OSKAR_IMAGE_TYPE_STOKES_I ||
            im_type == OSKAR_IMAGE_TYPE_PSF))
    {
        *status = OSKAR_ERR_SETTINGS_IMAGE;
        return 0;
    }

    /* Get precision of data type. */
    int type = oskar_mem_precision(oskar_vis_amplitude_const(vis));

    /* Create and set up the image cube. */
    im = oskar_set_up_image(vis, settings, status);
    if (!im || *status)
    {
        oskar_image_free(im, status);
        return 0;
    }

    /* Evaluate Stokes parameters (if required). */
    oskar_Mem* stokes = oskar_get_image_stokes(vis, settings, status);
    if (*status)
    {
        oskar_image_free(im, status);
        oskar_mem_free(stokes, status);
        return 0;
    }

    int num_vis = 0;
    if (settings->time_snapshots && settings->channel_snapshots)
        num_vis = num_baselines;
    else if (settings->time_snapshots && !settings->channel_snapshots)
        num_vis = num_baselines * num_channels;
    else if (!settings->time_snapshots && settings->channel_snapshots)
        num_vis = num_baselines * num_times;
    else /* Time and frequency synthesis. */
        num_vis = num_baselines * num_channels * num_times;
    oskar_Mem *uu_im = 0, *vv_im = 0, *ww_im = 0, *vis_im = 0;
    oskar_Mem *uu_tmp = 0, *vv_tmp = 0, *ww_tmp = 0, *l = 0, *m = 0;
    uu_im = oskar_mem_create(type, location, num_vis, status);
    vv_im = oskar_mem_create(type, location, num_vis, status);
    ww_im = oskar_mem_create(type, location, num_vis, status);
    vis_im = oskar_mem_create(type | OSKAR_COMPLEX, location, num_vis, status);
    if (settings->direction_type == OSKAR_IMAGE_DIRECTION_RA_DEC)
    {
        uu_tmp = oskar_mem_create(type, location, num_vis, status);
        vv_tmp = oskar_mem_create(type, location, num_vis, status);
        ww_tmp = oskar_mem_create(type, location, num_vis, status);
    }
    /* Allocate pixel coordinate grid required for the DFT imager. */
    l = oskar_mem_create(type, location, num_pixels, status);
    m = oskar_mem_create(type, location, num_pixels, status);
    if (!*status)
    {
        if (type == OSKAR_SINGLE)
        {
            oskar_evaluate_image_lm_grid_f(size, size, fov_rad, fov_rad,
                    oskar_mem_float(l, status), oskar_mem_float(m, status));
        }
        else
        {
            oskar_evaluate_image_lm_grid_d(size, size, fov_rad, fov_rad,
                    oskar_mem_double(l, status), oskar_mem_double(m, status));
        }
    }

    /* Get a pointer to the slice of the image cube being imaged. */
    oskar_Mem* im_slice = oskar_mem_create_alias(0, 0, num_pixels, status);

    /* If imaging away from the beam direction, evaluate l0-l, m0-m, n0-n
     * for the new pointing centre as well as a set of baseline coordinates
     * corresponding to the user specified imaging direction. */
    double delta_l = 0.0, delta_m = 0.0, delta_n = 0.0;
    oskar_Mem *uu_rot = 0, *vv_rot = 0, *ww_rot = 0;
    if (settings->direction_type == OSKAR_IMAGE_DIRECTION_RA_DEC)
    {
        double ra_rad = settings->ra_deg * DEG2RAD;
        double dec_rad = settings->dec_deg * DEG2RAD;
        double ra0_rad = oskar_vis_phase_centre_ra_deg(vis) * DEG2RAD;
        double dec0_rad = oskar_vis_phase_centre_dec_deg(vis) * DEG2RAD;
        double l1, m1, n1;
        int num_elements = num_baselines * num_times;

        oskar_convert_lon_lat_to_relative_directions_d(1,
                &ra_rad, &dec_rad, ra0_rad, dec0_rad, &l1, &m1, &n1);
        delta_l = 0 - l1;
        delta_m = 0 - m1;
        delta_n = 1 - n1;

        uu_rot = oskar_mem_create(type, location, num_elements, status);
        vv_rot = oskar_mem_create(type, location, num_elements, status);
        ww_rot = oskar_mem_create(type, location, num_elements, status);

        /* Work array for baseline evaluation. */
        oskar_Mem *work_uvw;
        work_uvw = oskar_mem_create(type, location, 3 * num_stations, status);
        oskar_convert_ecef_to_baseline_uvw(num_stations,
                station_ecef_x, station_ecef_y, station_ecef_z,
                ra_rad, dec_rad, num_times, oskar_vis_time_start_mjd_utc(vis),
                oskar_vis_time_inc_sec(vis) * SEC2DAYS, 0,
                uu_rot, vv_rot, ww_rot, work_uvw, status);
        oskar_mem_free(work_uvw, status);
    }

    /* Construct the image cube. */
    for (int i = 0, c = 0; c < im_num_chan; ++c)
    {
        if (*status) break;
        int vis_chan = im_chan_range[0] + c;
        double im_freq = oskar_image_freq_start_hz(im) +
                c * oskar_image_freq_inc_hz(im);
        oskar_log_message(log, 'M', 0, "Channel %3d/%d [%.4f MHz]",
                c + 1, im_num_chan, im_freq / 1e6);

        for (int t = 0; t < im_num_times; ++t)
        {
            if (*status) break;
            int vis_time = im_time_range[0] + t;

            /* Evaluate baseline coordinates needed for imaging. */
            if (settings->direction_type == OSKAR_IMAGE_DIRECTION_OBSERVATION)
            {
                *status = oskar_get_image_baseline_coords(uu_im, vv_im, ww_im,
                        baseline_uu, baseline_vv, baseline_ww, num_times,
                        num_baselines, num_channels, freq_start_hz,
                        freq_inc_hz, vis_time, im_freq, settings);
            }
            else if (settings->direction_type == OSKAR_IMAGE_DIRECTION_RA_DEC)
            {
                /* Rotated coordinates (used for imaging) */
                *status = oskar_get_image_baseline_coords(uu_im, vv_im, ww_im,
                        uu_rot, vv_rot, ww_rot, num_times,
                        num_baselines, num_channels, freq_start_hz,
                        freq_inc_hz, vis_time, im_freq, settings);

                /* Unrotated coordinates (used for phase rotation) */
                *status = oskar_get_image_baseline_coords(uu_tmp, vv_tmp, ww_tmp,
                        baseline_uu, baseline_vv, baseline_ww, num_times,
                        num_baselines, num_channels, freq_start_hz,
                        freq_inc_hz, vis_time, im_freq, settings);
            }
            else
            {
                *status = OSKAR_ERR_SETTINGS_IMAGE;
            }

            for (int p = 0; p < num_pols; ++p, ++i)
            {
                if (*status) break;

                oskar_log_message(log, 'M', 1, "Making image %3i/%i, "
                        "cube index (c=%i, t=%i, p=%i)",
                        i+1, (im_num_chan*im_num_times*num_pols), c, t, p);

                /* Get visibility amplitudes for imaging. */
                if (im_type == OSKAR_IMAGE_TYPE_PSF)
                {
                    oskar_mem_set_value_real(vis_im, 1.0, 0, 0, status);
                }
                else
                {
                    *status = oskar_get_image_vis_amps(vis_im, vis, stokes,
                            settings, vis_chan, vis_time, p);

                    /* Phase rotate the visibilities. */
                    if (settings->direction_type == OSKAR_IMAGE_DIRECTION_RA_DEC)
                    {
                        phase_rotate_vis_amps(vis_im, num_vis, type,
                                delta_l, delta_m, delta_n, uu_tmp, vv_tmp,
                                ww_tmp, im_freq);
                    }
                }

                oskar_log_message(log, 'M', 2, "Number of visibilities %i",
                        (int)oskar_mem_length(vis_im));

                /* Get pointer to slice of the image cube. */
                int slice_offset = ((c * im_num_times + t) * num_pols + p) * num_pixels;
                oskar_mem_set_alias(im_slice, oskar_image_data(im),
                        slice_offset, num_pixels, status);

                /* Make the image */
                if (settings->transform_type == OSKAR_IMAGE_DFT_2D)
                {
                    oskar_make_image_dft(im_slice, uu_im, vv_im, vis_im,
                            l, m, im_freq, status);
                }
                else
                {
                    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
                }
            }
        }
    }

    oskar_mem_free(stokes, status);
    oskar_mem_free(uu_rot, status);
    oskar_mem_free(vv_rot, status);
    oskar_mem_free(ww_rot, status);
    oskar_mem_free(uu_tmp, status);
    oskar_mem_free(vv_tmp, status);
    oskar_mem_free(ww_tmp, status);
    oskar_mem_free(uu_im, status);
    oskar_mem_free(vv_im, status);
    oskar_mem_free(ww_im, status);
    oskar_mem_free(vis_im, status);
    oskar_mem_free(l, status);
    oskar_mem_free(m, status);
    oskar_mem_free(im_slice, status);

    /* Return a handle to the image cube. */
    return im;
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
        const double* uu_ = (const double*)oskar_mem_void_const(uu);
        const double* vv_ = (const double*)oskar_mem_void_const(vv);
        const double* ww_ = (const double*)oskar_mem_void_const(ww);
        double2* amp_ = (double2*)oskar_mem_void(amps);

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
        const float* uu_ = (const float*)oskar_mem_void_const(uu);
        const float* vv_ = (const float*)oskar_mem_void_const(vv);
        const float* ww_ = (const float*)oskar_mem_void_const(ww);
        float2* amp_ = (float2*)oskar_mem_void(amps);

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
