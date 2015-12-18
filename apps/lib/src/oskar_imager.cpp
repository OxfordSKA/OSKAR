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

#include <cuda_runtime_api.h>
#include <oskar_cmath.h>
#include <oskar_convert_ecef_to_baseline_uvw.h>
#include <oskar_convert_lon_lat_to_relative_directions.h>
#include <oskar_evaluate_image_lm_grid.h>
#include <oskar_evaluate_image_ranges.h>
#include <oskar_get_error_string.h>
#include <oskar_get_image_baseline_coords.h>
#include <oskar_get_image_stokes.h>
#include <oskar_get_image_vis_amps.h>
#include <oskar_imager.h>
#include <oskar_image.h>
#include <oskar_log.h>
#include <oskar_make_image_dft.h>
#include <oskar_settings_log.h>
#include <oskar_settings_load.h>
#include <oskar_settings_old_free.h>
#include <oskar_vis.h>

#include <fitsio.h>
#include <stdio.h>
#include <string.h>

#include "fits/oskar_fits_image_write.h"

#define DEG2RAD M_PI/180.0

#define SEC2DAYS 1.15740740740740740740741e-5

#ifdef __cplusplus
extern "C" {
#endif

/* Memory allocated once, on the host. */
struct HostData
{
    oskar_Settings_old s;
    oskar_Mem *uu_im, *vv_im, *ww_im, *vis_im;
    oskar_Mem *uu_tmp, *vv_tmp, *ww_tmp, *l, *m, *stokes, *im_slice;
    oskar_Mem *uu_rot, *vv_rot, *ww_rot;
    oskar_Binary* vis_file;
    oskar_Vis* vis;
	oskar_Image* im;
	fitsfile* fits_file;
};
typedef struct HostData HostData;


static fitsfile* create_fits_file(const char* filename, int precision,
        int width, int height, int num_times, int num_channels,
        double centre_deg[2], double fov_deg[2], double start_time_mjd,
        double delta_time_sec, double start_freq_hz, double delta_freq_hz,
        const char* settings_log, size_t settings_log_length, int* status);
static void write_axis_header(fitsfile* fptr, int axis_id,
        const char* ctype, const char* ctype_comment, double crval,
        double cdelt, double crpix, double crota, int* status);
static void phase_rotate_vis_amps(oskar_Mem* amps, int num_vis, int type,
        double delta_l, double delta_m, double delta_n, const oskar_Mem* uu,
        const oskar_Mem* vv, const oskar_Mem* ww, double freq);
static void free_host_data(HostData* h, int* status);

static double fov_to_cellsize(double fov_deg, int num_pixels)
{
    double max, inc;
    max = sin(fov_deg * M_PI / 360.0); /* Divide by 2. */
    inc = max / (0.5 * num_pixels);
    return asin(inc) * 180.0 / M_PI;
}

void oskar_imager(const char* settings_file, oskar_Log* log, int* status)
{
    int im_chan_range[2], im_time_range[2];
    int vis_chan_range[2], vis_time_range[2];
    int im_num_chan = 1, im_num_times = 1, im_num_pols = 1;
    int num_vis = 0;
    HostData* h = 0;
    double delta_l = 0.0, delta_m = 0.0, delta_n = 0.0;
    const oskar_Mem *station_ecef_x, *station_ecef_y, *station_ecef_z;
    const oskar_Mem *baseline_uu, *baseline_vv, *baseline_ww;

    /* Create the host data structure (initialised with all bits zero). */
    h = (HostData*) calloc(1, sizeof(HostData));

    /* Load the settings file. */
    oskar_log_section(log, 'M', "Loading settings file '%s'", settings_file);
    oskar_settings_old_load(&h->s, log, settings_file, status);
    if (*status) { free_host_data(h, status); return; }

    /* Log the relevant settings. */
    const oskar_SettingsImage* settings = &h->s.image;
    oskar_log_set_keep_file(log, h->s.sim.keep_log_file);
    oskar_log_settings_image(log, &h->s);

    /* Get image settings. */
    double img_ra_deg = settings->ra_deg;
    double img_dec_deg = settings->dec_deg;
    double fov_deg = settings->fov_deg;
    double fov_rad = fov_deg * DEG2RAD;
    int size = settings->size;
    int num_pixels = size * size;
    int channel_snapshots = settings->channel_snapshots;
    int time_snapshots = settings->time_snapshots;
    int im_type = settings->image_type;
    int direction_type = settings->direction_type;
    int transform_type = settings->transform_type;
    const char* image_name = settings->fits_image;
    const char* vis_name = settings->input_vis_data;

    /* Check filenames have been set. */
    if (!image_name)
    {
        oskar_log_error(log, "No output image file specified.");
        free_host_data(h, status);
        *status = OSKAR_ERR_SETTINGS_IMAGE;
        return;
    }
    if (!vis_name)
    {
        oskar_log_error(log, "No input visibility data file specified.");
        free_host_data(h, status);
        *status = OSKAR_ERR_SETTINGS_IMAGE;
        return;
    }

    /* Get number of output polarisations from image type. */
    if (im_type == OSKAR_IMAGE_TYPE_STOKES ||
    		im_type == OSKAR_IMAGE_TYPE_POL_LINEAR)
    	im_num_pols = 4;

    /* Read the visibility data. */
    h->vis_file = oskar_binary_create(vis_name, 'r', status);
    h->vis = oskar_vis_read(h->vis_file, status);
    if (*status)
    {
        oskar_log_error(log, "Failed to read visibility data (%s).",
                oskar_get_error_string(*status));
        free_host_data(h, status);
        return;
    }
    int prec = oskar_mem_precision(oskar_vis_amplitude_const(h->vis));
    int vis_num_pols = oskar_vis_num_pols(h->vis);
    int vis_num_channels = oskar_vis_num_channels(h->vis);
    int vis_num_times = oskar_vis_num_times(h->vis);
    int num_stations = oskar_vis_num_stations(h->vis);
    int num_baselines = oskar_vis_num_baselines(h->vis);
    double freq_start_hz = oskar_vis_freq_start_hz(h->vis);
    double freq_inc_hz = oskar_vis_freq_inc_hz(h->vis);
    double time_start_mjd_utc = oskar_vis_time_start_mjd_utc(h->vis);
    double time_inc_sec = oskar_vis_time_inc_sec(h->vis);
    double vis_ra_deg = oskar_vis_phase_centre_ra_deg(h->vis);
    double vis_dec_deg = oskar_vis_phase_centre_dec_deg(h->vis);
    station_ecef_x = oskar_vis_station_x_offset_ecef_metres_const(h->vis);
    station_ecef_y = oskar_vis_station_y_offset_ecef_metres_const(h->vis);
    station_ecef_z = oskar_vis_station_z_offset_ecef_metres_const(h->vis);
    baseline_uu = oskar_vis_baseline_uu_metres_const(h->vis);
    baseline_vv = oskar_vis_baseline_vv_metres_const(h->vis);
    baseline_ww = oskar_vis_baseline_ww_metres_const(h->vis);

    /* Make the image. */
    oskar_log_section(log, 'M', "Starting OSKAR imager...");

    if (direction_type == OSKAR_IMAGE_DIRECTION_OBSERVATION)
    {
    	img_ra_deg = vis_ra_deg;
    	img_dec_deg = vis_dec_deg;
    }

    /* Set the channel and time range for the image cube [output range]. */
    oskar_evaluate_image_range(im_chan_range, channel_snapshots,
            settings->channel_range, vis_num_channels, status);
    im_num_chan = im_chan_range[1] - im_chan_range[0] + 1;
    oskar_evaluate_image_range(im_time_range, time_snapshots,
            settings->time_range, vis_num_times, status);
    im_num_times = im_time_range[1] - im_time_range[0] + 1;
    if (*status) { free_host_data(h, status); return; }

    if (im_num_times > vis_num_times || im_num_chan > vis_num_channels ||
            im_num_pols > vis_num_pols)
    {
        free_host_data(h, status);
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    if (vis_num_pols == 1 && !(im_type == OSKAR_IMAGE_TYPE_STOKES_I ||
            im_type == OSKAR_IMAGE_TYPE_PSF))
    {
        oskar_log_error(log, "Incompatible image type selected for scalar "
                "(Stokes-I) visibility data.");
        free_host_data(h, status);
        *status = OSKAR_ERR_SETTINGS_IMAGE;
        return;
    }

    /* Time and channel range for data. */
    oskar_evaluate_image_data_range(vis_chan_range, settings->channel_range,
            vis_num_channels, status);
    oskar_evaluate_image_data_range(vis_time_range, settings->time_range,
            vis_num_times, status);

    /* Create the image and size it. */
    h->im = oskar_image_create(prec, OSKAR_CPU, status);
    oskar_image_resize(h->im, size, size,
    		im_num_pols, im_num_times, im_num_chan, status);
    if (*status) { free_host_data(h, status); return; }
    oskar_image_set_fov(h->im, fov_deg, fov_deg);
    oskar_image_set_centre(h->im, img_ra_deg, img_dec_deg);
    oskar_image_set_time(h->im, time_start_mjd_utc +
            (vis_time_range[0] * time_inc_sec * SEC2DAYS),
            time_snapshots ? time_inc_sec : 0.0);
    if (channel_snapshots)
    {
        oskar_image_set_freq(h->im, freq_start_hz +
                vis_chan_range[0] * freq_inc_hz, freq_inc_hz);
    }
    else
    {
        double chan0 = (vis_chan_range[1] - vis_chan_range[0]) / 2.0;
        oskar_image_set_freq(h->im, freq_start_hz +
                chan0 * freq_inc_hz, 0.0);
    }
    oskar_image_set_coord_frame(h->im, OSKAR_IMAGE_COORD_FRAME_EQUATORIAL);

    /* Evaluate Stokes parameters. */
    h->stokes = oskar_get_image_stokes(h->vis, settings, status);
    if (*status) { free_host_data(h, status); return; }

    if (time_snapshots && channel_snapshots)
        num_vis = num_baselines;
    else if (time_snapshots && !channel_snapshots)
        num_vis = num_baselines * vis_num_channels;
    else if (!time_snapshots && channel_snapshots)
        num_vis = num_baselines * vis_num_times;
    else /* Time and frequency synthesis. */
        num_vis = num_baselines * vis_num_channels * vis_num_times;
    h->uu_im = oskar_mem_create(prec, OSKAR_CPU, num_vis, status);
    h->vv_im = oskar_mem_create(prec, OSKAR_CPU, num_vis, status);
    h->ww_im = oskar_mem_create(prec, OSKAR_CPU, num_vis, status);
    h->vis_im = oskar_mem_create(prec | OSKAR_COMPLEX, OSKAR_CPU, num_vis, status);
    if (direction_type == OSKAR_IMAGE_DIRECTION_RA_DEC)
    {
        h->uu_tmp = oskar_mem_create(prec, OSKAR_CPU, num_vis, status);
        h->vv_tmp = oskar_mem_create(prec, OSKAR_CPU, num_vis, status);
        h->ww_tmp = oskar_mem_create(prec, OSKAR_CPU, num_vis, status);
    }

    /* Calculate pixel coordinate grid required for the DFT imager. */
    h->l = oskar_mem_create(prec, OSKAR_CPU, num_pixels, status);
    h->m = oskar_mem_create(prec, OSKAR_CPU, num_pixels, status);
    if (!*status)
    {
        if (prec == OSKAR_SINGLE)
            oskar_evaluate_image_lm_grid_f(size, size, fov_rad, fov_rad,
                    oskar_mem_float(h->l, status),
					oskar_mem_float(h->m, status));
        else
            oskar_evaluate_image_lm_grid_d(size, size, fov_rad, fov_rad,
                    oskar_mem_double(h->l, status),
					oskar_mem_double(h->m, status));
    }

    /* Get a pointer to the slice of the image cube being imaged. */
    h->im_slice = oskar_mem_create_alias(0, 0, num_pixels, status);

    /* If imaging away from the beam direction, evaluate l0-l, m0-m, n0-n
     * for the new pointing centre as well as a set of baseline coordinates
     * corresponding to the user specified imaging direction. */
    if (direction_type == OSKAR_IMAGE_DIRECTION_RA_DEC)
    {
        oskar_Mem *work_uvw;
        double l1, m1, n1, ra_rad, dec_rad, ra0_rad, dec0_rad;
        int num_elements;

        ra_rad = img_ra_deg * DEG2RAD;
        dec_rad = img_dec_deg * DEG2RAD;
        ra0_rad = vis_ra_deg * DEG2RAD;
        dec0_rad = vis_dec_deg * DEG2RAD;
        num_elements = num_baselines * vis_num_times;

        oskar_convert_lon_lat_to_relative_directions_d(1,
                &ra_rad, &dec_rad, ra0_rad, dec0_rad, &l1, &m1, &n1);
        delta_l = 0 - l1;
        delta_m = 0 - m1;
        delta_n = 1 - n1;

        h->uu_rot = oskar_mem_create(prec, OSKAR_CPU, num_elements, status);
        h->vv_rot = oskar_mem_create(prec, OSKAR_CPU, num_elements, status);
        h->ww_rot = oskar_mem_create(prec, OSKAR_CPU, num_elements, status);

        /* Work array for baseline evaluation. */
        work_uvw = oskar_mem_create(prec, OSKAR_CPU, 3 * num_stations, status);
        oskar_convert_ecef_to_baseline_uvw(num_stations,
                station_ecef_x, station_ecef_y, station_ecef_z,
                ra_rad, dec_rad, vis_num_times, time_start_mjd_utc,
                time_inc_sec * SEC2DAYS, 0,
				h->uu_rot, h->vv_rot, h->ww_rot, work_uvw, status);
        oskar_mem_free(work_uvw, status);
    }

    /* Construct the image cube. */
    for (int i = 0, c = 0; c < im_num_chan; ++c)
    {
        if (*status) break;
        int vis_chan = im_chan_range[0] + c;
        double im_freq = oskar_image_freq_start_hz(h->im) +
                c * oskar_image_freq_inc_hz(h->im);
        oskar_log_message(log, 'M', 0, "Channel %3d/%d [%.4f MHz]",
                c + 1, im_num_chan, im_freq / 1e6);

        for (int t = 0; t < im_num_times; ++t)
        {
            if (*status) break;
            int vis_time = im_time_range[0] + t;

            /* Evaluate baseline coordinates needed for imaging. */
            if (direction_type == OSKAR_IMAGE_DIRECTION_RA_DEC)
            {
                /* Rotated coordinates (used for imaging) */
                *status = oskar_get_image_baseline_coords(
                		h->uu_im, h->vv_im, h->ww_im,
                		h->uu_rot, h->vv_rot, h->ww_rot, vis_num_times,
                        num_baselines, vis_num_channels, freq_start_hz,
                        freq_inc_hz, vis_time, im_freq, settings);

                /* Unrotated coordinates (used for phase rotation) */
                *status = oskar_get_image_baseline_coords(
                		h->uu_tmp, h->vv_tmp, h->ww_tmp,
                        baseline_uu, baseline_vv, baseline_ww, vis_num_times,
                        num_baselines, vis_num_channels, freq_start_hz,
                        freq_inc_hz, vis_time, im_freq, settings);
            }
            else
            {
                *status = oskar_get_image_baseline_coords(
                		h->uu_im, h->vv_im, h->ww_im,
                        baseline_uu, baseline_vv, baseline_ww, vis_num_times,
                        num_baselines, vis_num_channels, freq_start_hz,
                        freq_inc_hz, vis_time, im_freq, settings);
            }

            for (int p = 0; p < im_num_pols; ++p, ++i)
            {
                if (*status) break;

                oskar_log_message(log, 'M', 1, "Making image %3i/%i, "
                        "cube index (c=%i, t=%i, p=%i)",
                        i+1, (im_num_chan*im_num_times*im_num_pols), c, t, p);

                /* Get visibility amplitudes for imaging. */
                if (im_type == OSKAR_IMAGE_TYPE_PSF)
                    oskar_mem_set_value_real(h->vis_im, 1.0, 0, 0, status);
                else
                {
                    *status = oskar_get_image_vis_amps(h->vis_im, h->vis,
                    		h->stokes, settings, vis_chan, vis_time, p);

                    /* Phase rotate the visibilities. */
                    if (direction_type == OSKAR_IMAGE_DIRECTION_RA_DEC)
                        phase_rotate_vis_amps(h->vis_im, num_vis, prec,
                                delta_l, delta_m, delta_n,
								h->uu_tmp, h->vv_tmp, h->ww_tmp, im_freq);
                }

                /* Get pointer to slice of the image cube. */
                int slice_offset = ((c * im_num_times + t) * im_num_pols + p) * num_pixels;
                oskar_mem_set_alias(h->im_slice, oskar_image_data(h->im),
                        slice_offset, num_pixels, status);

                /* Make the image */
                if (transform_type == OSKAR_IMAGE_DFT_2D)
                    oskar_make_image_dft(h->im_slice, h->uu_im, h->vv_im,
                    		h->vis_im, h->l, h->m, im_freq, status);
                else
                    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
            }
        }
    }

    /* Write the image to file. */
    if (image_name && !*status)
    {
        oskar_log_message(log, 'M', 0, "Writing FITS image file: '%s'", image_name);
        oskar_fits_image_write(h->im, log, image_name, status);
    }

    if (!*status)
        oskar_log_section(log, 'M', "Run complete.");
    free_host_data(h, status);
    cudaDeviceReset();
}


static fitsfile* create_fits_file(const char* filename, int precision,
        int width, int height, int num_times, int num_channels,
        double centre_deg[2], double fov_deg[2], double start_time_mjd,
        double delta_time_sec, double start_freq_hz, double delta_freq_hz,
        const char* settings_log, size_t settings_log_length, int* status)
{
    int imagetype;
    long naxes[4], naxes_dummy[4] = {1l, 1l, 1l, 1l};
    double delta;
    fitsfile* f = 0;
    FILE* t = 0;
    const char* line;
    size_t length;
    if (*status) return 0;

    /* Create a new FITS file and write the image headers. */
    t = fopen(filename, "rb");
    if (t)
    {
        fclose(t);
        remove(filename);
    }
    imagetype = (precision == OSKAR_DOUBLE ? DOUBLE_IMG : FLOAT_IMG);
    naxes[0]  = width;
    naxes[1]  = height;
    naxes[2]  = num_channels;
    naxes[3]  = num_times;
    fits_create_file(&f, filename, status);
    fits_create_img(f, imagetype, 4, naxes_dummy, status);
    fits_write_date(f, status);

    /* Write axis headers. */
    delta = fov_to_cellsize(fov_deg[0], width);
    write_axis_header(f, 1, "RA---SIN", "Right Ascension",
    		centre_deg[0], -delta, (width + 1) / 2.0, 0.0, status);
    delta = fov_to_cellsize(fov_deg[1], height);
    write_axis_header(f, 2, "DEC--SIN", "Declination",
    		centre_deg[1], delta, (height + 1) / 2.0, 0.0, status);
    write_axis_header(f, 3, "FREQ", "Frequency",
    		start_freq_hz, delta_freq_hz, 1.0, 0.0, status);
    write_axis_header(f, 4, "UTC", "Time",
    		start_time_mjd, delta_time_sec, 1.0, 0.0, status);

    /* Write other headers. */
    fits_write_key_str(f, "TIMESYS", "UTC", NULL, status);
    fits_write_key_str(f, "TIMEUNIT", "s", "Time axis units", status);
    fits_write_key_dbl(f, "MJD-OBS", start_time_mjd, 10, "Start time", status);
    fits_write_key_dbl(f, "OBSRA", centre_deg[0], 10, "RA", status);
    fits_write_key_dbl(f, "OBSDEC", centre_deg[1], 10, "DEC", status);

    /* Write the settings log up to this point as HISTORY comments. */
    line = settings_log;
    length = settings_log_length;
    for (;;)
    {
        const char* eol;
        fits_write_history(f, line, status);
        eol = (const char*) memchr(line, '\0', length);
        if (!eol) break;
        eol += 1;
        length -= (eol - line);
        line = eol;
    }

    /* Update header keywords with the correct axis lengths.
     * Needs to be done here because CFITSIO doesn't let us write only the
     * file header with the correct axis lengths to start with. This trick
     * allows us to create a small dummy image block to write only the headers,
     * and not waste effort moving a huge block of zeros within the file. */
    fits_update_key_lng(f, "NAXIS1", naxes[0], 0, status);
    fits_update_key_lng(f, "NAXIS2", naxes[1], 0, status);
    fits_update_key_lng(f, "NAXIS3", naxes[2], 0, status);
    fits_update_key_lng(f, "NAXIS4", naxes[3], 0, status);

    return f;
}


void write_axis_header(fitsfile* fptr, int axis_id,
        const char* ctype, const char* ctype_comment, double crval,
        double cdelt, double crpix, double crota, int* status)
{
    char key[FLEN_KEYWORD], value[FLEN_VALUE], comment[FLEN_COMMENT];
    int decimals = 10;
    if (*status) return;

    strncpy(comment, ctype_comment, FLEN_COMMENT-1);
    strncpy(value, ctype, FLEN_VALUE-1);
    fits_make_keyn("CTYPE", axis_id, key, status);
    fits_write_key_str(fptr, key, value, comment, status);
    fits_make_keyn("CRVAL", axis_id, key, status);
    fits_write_key_dbl(fptr, key, crval, decimals, NULL, status);
    fits_make_keyn("CDELT", axis_id, key, status);
    fits_write_key_dbl(fptr, key, cdelt, decimals, NULL, status);
    fits_make_keyn("CRPIX", axis_id, key, status);
    fits_write_key_dbl(fptr, key, crpix, decimals, NULL, status);
    fits_make_keyn("CROTA", axis_id, key, status);
    fits_write_key_dbl(fptr, key, crota, decimals, NULL, status);
}


/*
 * TODO Make this a stand-alone function?
 *
 * Note that the coordinates uu, vv, ww are in metres.
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

    if (type == OSKAR_DOUBLE)
    {
    	const double *uu_, *vv_, *ww_;
    	double2* amp_;
        uu_ = (const double*)oskar_mem_void_const(uu);
        vv_ = (const double*)oskar_mem_void_const(vv);
        ww_ = (const double*)oskar_mem_void_const(ww);
        amp_ = (double2*)oskar_mem_void(amps);

        for (i = 0; i < num_vis; ++i)
        {
        	double u, v, w, arg, phase_re, phase_im, re, im;
            u = uu_[i] * inv_lambda;
            v = vv_[i] * inv_lambda;
            w = ww_[i] * inv_lambda;
            arg = 2.0 * M_PI * (u * delta_l + v * delta_m + w * delta_n);
            phase_re = cos(arg);
            phase_im = sin(arg);
            re = amp_[i].x * phase_re - amp_[i].y * phase_im;
            im = amp_[i].x * phase_im + amp_[i].y * phase_re;
            amp_[i].x = re;
            amp_[i].y = im;
        }
    }
    else
    {
    	const float *uu_, *vv_, *ww_;
    	float2* amp_;
        uu_ = (const float*)oskar_mem_void_const(uu);
        vv_ = (const float*)oskar_mem_void_const(vv);
        ww_ = (const float*)oskar_mem_void_const(ww);
        amp_ = (float2*)oskar_mem_void(amps);

        for (i = 0; i < num_vis; ++i)
        {
        	float u, v, w, arg, phase_re, phase_im, re, im;
            u = uu_[i] * inv_lambda;
            v = vv_[i] * inv_lambda;
            w = ww_[i] * inv_lambda;
            arg = 2.0 * M_PI * (u * delta_l + v * delta_m + w * delta_n);
            phase_re = cos(arg);
            phase_im = sin(arg);
            re = amp_[i].x * phase_re - amp_[i].y * phase_im;
            im = amp_[i].x * phase_im + amp_[i].y * phase_re;
            amp_[i].x = re;
            amp_[i].y = im;
        }
    }
}


static void free_host_data(HostData* h, int* status)
{
    oskar_mem_free(h->stokes, status);
    oskar_mem_free(h->uu_rot, status);
    oskar_mem_free(h->vv_rot, status);
    oskar_mem_free(h->ww_rot, status);
    oskar_mem_free(h->uu_tmp, status);
    oskar_mem_free(h->vv_tmp, status);
    oskar_mem_free(h->ww_tmp, status);
    oskar_mem_free(h->uu_im, status);
    oskar_mem_free(h->vv_im, status);
    oskar_mem_free(h->ww_im, status);
    oskar_mem_free(h->vis_im, status);
    oskar_mem_free(h->l, status);
    oskar_mem_free(h->m, status);
    oskar_mem_free(h->im_slice, status);
    oskar_binary_free(h->vis_file);
    oskar_vis_free(h->vis, status);
    oskar_image_free(h->im, status);
    oskar_settings_old_free(&h->s);
    free(h);
}

#ifdef __cplusplus
}
#endif
