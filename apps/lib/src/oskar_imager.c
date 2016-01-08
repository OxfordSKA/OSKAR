/*
 * Copyright (c) 2012-2016, The University of Oxford
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
#include <oskar_dft_c2r_2d_cuda.h>
#include <oskar_evaluate_image_lmn_grid.h>
#include <oskar_evaluate_image_ranges.h>
#include <oskar_get_error_string.h>
#include <oskar_imager.h>
#include <oskar_image.h>
#include <oskar_log.h>
#include <oskar_make_image_dft.h>
#include <oskar_settings_log.h>
#include <oskar_settings_load.h>
#include <oskar_settings_old_free.h>
#include <oskar_vis.h>
#include <oskar_vis_header.h>
#include <oskar_vis_block.h>

#include <fitsio.h>
#include <stdio.h>
#include <string.h>

#include "fits/oskar_fits_image_write.h"

#define DEG2RAD M_PI/180.0

#define SEC2DAYS 1.15740740740740740740741e-5

#ifdef __cplusplus
extern "C" {
#endif

/* Memory allocated per GPU. */
struct DeviceData
{
    oskar_Mem *l, *m, *n; /* Direction cosines for DFT imager. */
    oskar_Mem *uu, *vv, *ww, *amp;
    oskar_Mem *slice_gpu, *slice_cpu;
};
typedef struct DeviceData DeviceData;

/* Memory allocated once, on the host. */
struct HostData
{
    /* Data from settings. */
    int prec, num_gpus, cuda_device_ids[1];
    int channel_range[2], time_range[2];
    int channel_snapshots, time_snapshots;
    int im_chan_range[2], im_time_range[2];

    int vis_chan_range[2], vis_time_range[2];
    int num_stations, num_baselines, use_stokes;
    int im_num_chan, im_num_times, im_num_pols;
    int vis_num_channels, vis_num_times, vis_num_pols;
    int size, num_pixels, im_type, transform_type, direction_type, num_vis;
    double im_ra_deg, im_dec_deg, vis_ra_deg, vis_dec_deg;
    double vis_freq_start_hz, im_freq_start_hz, freq_inc_hz;
    double vis_time_start_mjd_utc, im_time_start_mjd_utc, time_inc_sec;
    double delta_l, delta_m, delta_n, fov_deg, fov_rad;
    char *image_name, *vis_name;
    oskar_Settings_old s;
    oskar_Mem *uu_im, *vv_im, *ww_im, *vis_im;
    oskar_Mem *uu_tmp, *vv_tmp, *ww_tmp, *l, *m, *n, *im_slice, *stokes;
    oskar_Mem *uu_rot, *vv_rot, *ww_rot, *work_uvw;
    const oskar_Mem *st_ecef_x, *st_ecef_y, *st_ecef_z;
    const oskar_Mem *baseline_uu, *baseline_vv, *baseline_ww;
    oskar_Binary* vis_file;
    oskar_Vis* vis; /* FIXME Remove. */
    oskar_VisHeader* hdr;
    oskar_VisBlock* blk;
    oskar_Image* im; /* FIXME Remove. */
    fitsfile* fits_file[4];
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
static void set_up_host_data(HostData* h, oskar_Log* log, int *status);
static void set_up_device_data(DeviceData* d, const HostData* h, int* status);
static void free_host_data(HostData* h, int* status);
static void free_device_data(int num_gpus, int* cuda_device_ids,
        DeviceData* d, int* status);
static void get_baseline_coords(int num_baselines, const oskar_Mem* vis_uu,
        const oskar_Mem* vis_vv, const oskar_Mem* vis_ww,
        const int vis_time_range[2], const int vis_chan_range[2], int vis_time,
        double freq_start_hz, double freq_inc_hz, double im_freq,
        int time_snapshots, int channel_snapshots,
        oskar_Mem* uu, oskar_Mem* vv, oskar_Mem* ww, int* status);
static void get_stokes(const oskar_Mem* in, oskar_Mem* out, int* status);
static void get_vis_amps(HostData* h, oskar_Mem* amps, const oskar_Mem* vis_in,
        int num_times, int vis_channel, int vis_time, int vis_pol, int* status);

static double fov_to_cellsize(double fov_deg, int num_pixels)
{
    double max, inc;
    max = sin(fov_deg * M_PI / 360.0); /* Divide by 2. */
    inc = max / (0.5 * num_pixels);
    return asin(inc) * 180.0 / M_PI;
}

void oskar_imager(const char* settings_file, oskar_Log* log, int* status)
{
    int i, c, t, p;
    DeviceData* d = 0;
    HostData* h = 0;

    /* Create the host data structure (initialised with all bits zero). */
    h = (HostData*) calloc(1, sizeof(HostData));

    /* Load the settings file. */
    oskar_log_section(log, 'M', "Loading settings file '%s'", settings_file);
    oskar_settings_old_load(&h->s, log, settings_file, status);
    if (*status) { free_host_data(h, status); return; }

    /* Log the relevant settings. */
    oskar_log_set_keep_file(log, h->s.sim.keep_log_file);
    oskar_log_settings_image(log, &h->s);

    /* Set up host data and check for errors. */
    set_up_host_data(h, log, status);
    if (*status) { free_host_data(h, status); return; }

    /* Initialise each of the requested GPUs and set up per-GPU memory. */
    d = (DeviceData*) calloc(h->num_gpus, sizeof(DeviceData));
    for (i = 0; i < h->num_gpus; ++i)
    {
        *status = (int) cudaSetDevice(h->cuda_device_ids[i]);
        if (*status)
        {
            free_device_data(h->num_gpus, h->cuda_device_ids, d, status);
            free_host_data(h, status);
            return;
        }
        set_up_device_data(&d[i], h, status);
    }

    /* Convert linear polarisations to Stokes parameters if required. */
    if (h->use_stokes)
        get_stokes(oskar_vis_amplitude_const(h->vis), h->stokes, status);

    /* Make the image. */
    oskar_log_section(log, 'M', "Starting imager...");

    for (i = 0, c = 0; c < h->im_num_chan; ++c)
    {
        int vis_chan, vis_time, slice_offset;
        double im_freq;

        if (*status) break;
        vis_chan = h->im_chan_range[0] + c;
        im_freq = h->im_freq_start_hz + c * h->freq_inc_hz;

        for (t = 0; t < h->im_num_times; ++t)
        {
            if (*status) break;
            vis_time = h->im_time_range[0] + t;

            /* Evaluate baseline coordinates needed for imaging. */
            if (h->direction_type == OSKAR_IMAGE_DIRECTION_RA_DEC)
            {
                /* Rotated coordinates (used for imaging) */
                get_baseline_coords(h->num_baselines, h->uu_rot,
                        h->vv_rot, h->ww_rot, h->vis_time_range,
                        h->vis_chan_range, vis_time, h->vis_freq_start_hz,
                        h->freq_inc_hz, im_freq, h->time_snapshots,
                        h->channel_snapshots, h->uu_im, h->vv_im, h->ww_im,
                        status);

                /* Unrotated coordinates (used for phase rotation) */
                get_baseline_coords(h->num_baselines, h->baseline_uu,
                        h->baseline_vv, h->baseline_ww, h->vis_time_range,
                        h->vis_chan_range, vis_time, h->vis_freq_start_hz,
                        h->freq_inc_hz, im_freq, h->time_snapshots,
                        h->channel_snapshots, h->uu_tmp, h->vv_tmp, h->ww_tmp,
                        status);
            }
            else
            {
                get_baseline_coords(h->num_baselines, h->baseline_uu,
                        h->baseline_vv, h->baseline_ww, h->vis_time_range,
                        h->vis_chan_range, vis_time, h->vis_freq_start_hz,
                        h->freq_inc_hz, im_freq, h->time_snapshots,
                        h->channel_snapshots, h->uu_im, h->vv_im, h->ww_im,
                        status);
            }

            for (p = 0; p < h->im_num_pols; ++p, ++i)
            {
                if (*status) break;

                /* Get visibility amplitudes for imaging. */
                if (h->im_type == OSKAR_IMAGE_TYPE_PSF)
                    oskar_mem_set_value_real(h->vis_im, 1.0, 0, 0, status);
                else
                {
                    get_vis_amps(h, h->vis_im,
                            (h->use_stokes ? h->stokes :
                                    oskar_vis_amplitude_const(h->vis)),
                            h->vis_num_times, vis_chan, vis_time, p, status);

                    /* Phase rotate the visibilities if required. */
                    if (h->direction_type == OSKAR_IMAGE_DIRECTION_RA_DEC)
                        phase_rotate_vis_amps(h->vis_im, h->num_vis, h->prec,
                                h->delta_l, h->delta_m, h->delta_n,
                                h->uu_tmp, h->vv_tmp, h->ww_tmp, im_freq);
                }

                /* Get pointer to slice of the image cube. */
                slice_offset = h->num_pixels *
                        ((c * h->im_num_times + t) * h->im_num_pols + p);
                oskar_mem_set_alias(h->im_slice, oskar_image_data(h->im),
                        slice_offset, h->num_pixels, status);

                /* Make the image */
                if (h->transform_type == OSKAR_IMAGE_DFT_2D)
                {
                    oskar_make_image_dft(h->im_slice, h->uu_im, h->vv_im,
                            h->vis_im, d[0].l, d[0].m, im_freq, status);
                }
                else
                    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
            }
        }
    }

    /* Write the image to file. */
    if (h->image_name && !*status)
    {
        oskar_log_message(log, 'M', 0, "Writing FITS image file: '%s'",
                h->image_name);
        oskar_fits_image_write(h->im, log, h->image_name, status);
    }

    if (!*status)
        oskar_log_section(log, 'M', "Run complete.");
    free_device_data(h->num_gpus, h->cuda_device_ids, d, status);
    free_host_data(h, status);
}


/*
 * Note that the coordinates uu, vv, ww are in metres.
 *
 * Ref:
 * Cornwell, T.J., & Perley, R.A., 1992,
 * "Radio-interferometric imaging of very large fields"
 */
void phase_rotate_vis_amps(oskar_Mem* amps, int num_vis, int type,
        double delta_l, double delta_m, double delta_n, const oskar_Mem* uu,
        const oskar_Mem* vv, const oskar_Mem* ww, double freq)
{
    int i;
    double scale_wavenumber = 2.0 * M_PI * freq / 299792458.0;

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
            u = uu_[i] * scale_wavenumber;
            v = vv_[i] * scale_wavenumber;
            w = ww_[i] * scale_wavenumber;
            arg = u * delta_l + v * delta_m + w * delta_n;
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
            u = uu_[i] * scale_wavenumber;
            v = vv_[i] * scale_wavenumber;
            w = ww_[i] * scale_wavenumber;
            arg = u * delta_l + v * delta_m + w * delta_n;
            phase_re = cos(arg);
            phase_im = sin(arg);
            re = amp_[i].x * phase_re - amp_[i].y * phase_im;
            im = amp_[i].x * phase_im + amp_[i].y * phase_re;
            amp_[i].x = re;
            amp_[i].y = im;
        }
    }
}



void set_up_host_data(HostData* h, oskar_Log* log, int *status)
{
    int amp_type;

    /* Get settings. */
    const oskar_SettingsImage* settings = &h->s.image;
    h->fov_deg = settings->fov_deg;
    h->fov_rad = h->fov_deg * DEG2RAD;
    h->im_ra_deg = settings->ra_deg;
    h->im_dec_deg = settings->dec_deg;
    h->num_gpus = 1;
    h->channel_range[0] = settings->channel_range[0];
    h->channel_range[1] = settings->channel_range[1];
    h->time_range[0] = settings->time_range[0];
    h->time_range[1] = settings->time_range[1];
    h->size = settings->size;
    h->num_pixels = h->size * h->size;
    h->channel_snapshots = settings->channel_snapshots;
    h->time_snapshots = settings->time_snapshots;
    h->im_type = settings->image_type;
    h->direction_type = settings->direction_type;
    h->transform_type = settings->transform_type;
    h->image_name = settings->fits_image;
    h->vis_name = settings->input_vis_data;
    h->use_stokes = (h->im_type == OSKAR_IMAGE_TYPE_STOKES ||
            h->im_type == OSKAR_IMAGE_TYPE_STOKES_I ||
            h->im_type == OSKAR_IMAGE_TYPE_STOKES_Q ||
            h->im_type == OSKAR_IMAGE_TYPE_STOKES_U ||
            h->im_type == OSKAR_IMAGE_TYPE_STOKES_V);

    /* Check filenames have been set. */
    if (!h->image_name)
    {
        oskar_log_error(log, "No output image file specified.");
        *status = OSKAR_ERR_SETTINGS_IMAGE;
        return;
    }
    if (!h->vis_name)
    {
        oskar_log_error(log, "No input visibility data file specified.");
        *status = OSKAR_ERR_SETTINGS_IMAGE;
        return;
    }

    /* Get number of output polarisations from image type. */
    if (h->im_type == OSKAR_IMAGE_TYPE_STOKES ||
            h->im_type == OSKAR_IMAGE_TYPE_POL_LINEAR)
        h->im_num_pols = 4;

    /* Read the visibility header data. */
    h->vis_file = oskar_binary_create(h->vis_name, 'r', status);
    h->hdr = oskar_vis_header_read(h->vis_file, status);
    h->vis = oskar_vis_read(h->vis_file, status); /* TODO Remove old vis. */
    if (*status)
    {
        oskar_log_error(log, "Failed to read visibility data (%s).",
                oskar_get_error_string(*status));
        return;
    }
    amp_type = oskar_vis_header_amp_type(h->hdr);
    h->prec = oskar_type_precision(amp_type);
    h->vis_num_pols = oskar_type_is_matrix(amp_type) ? 4 : 1;
    h->vis_num_channels = oskar_vis_header_num_channels_total(h->hdr);
    h->vis_num_times = oskar_vis_header_num_times_total(h->hdr);
    h->num_stations = oskar_vis_header_num_stations(h->hdr);
    h->num_baselines = h->num_stations * (h->num_stations - 1) / 2;
    h->vis_freq_start_hz = oskar_vis_header_freq_start_hz(h->hdr);
    h->freq_inc_hz = oskar_vis_header_freq_inc_hz(h->hdr);
    h->vis_time_start_mjd_utc = oskar_vis_header_time_start_mjd_utc(h->hdr);
    h->time_inc_sec = oskar_vis_header_time_inc_sec(h->hdr);
    h->vis_ra_deg = oskar_vis_header_phase_centre_ra_deg(h->hdr);
    h->vis_dec_deg = oskar_vis_header_phase_centre_dec_deg(h->hdr);
    h->st_ecef_x = oskar_vis_header_station_x_offset_ecef_metres_const(h->hdr);
    h->st_ecef_y = oskar_vis_header_station_y_offset_ecef_metres_const(h->hdr);
    h->st_ecef_z = oskar_vis_header_station_z_offset_ecef_metres_const(h->hdr);
    h->baseline_uu = oskar_vis_baseline_uu_metres_const(h->vis);
    h->baseline_vv = oskar_vis_baseline_vv_metres_const(h->vis);
    h->baseline_ww = oskar_vis_baseline_ww_metres_const(h->vis);

    if (h->direction_type == OSKAR_IMAGE_DIRECTION_OBSERVATION)
    {
        h->im_ra_deg = h->vis_ra_deg;
        h->im_dec_deg = h->vis_dec_deg;
    }

    /* Time and channel range for data. */
    oskar_evaluate_image_data_range(h->vis_chan_range, h->channel_range,
            h->vis_num_channels, status);
    oskar_evaluate_image_data_range(h->vis_time_range, h->time_range,
            h->vis_num_times, status);

    /* Time and channel range for image cube [output range]. */
    oskar_evaluate_image_range(h->im_time_range, h->time_snapshots,
            h->time_range, h->vis_num_times, status);
    oskar_evaluate_image_range(h->im_chan_range, h->channel_snapshots,
            h->channel_range, h->vis_num_channels, status);
    h->im_num_times = h->im_time_range[1] - h->im_time_range[0] + 1;
    h->im_num_chan = h->im_chan_range[1] - h->im_chan_range[0] + 1;
    h->im_time_start_mjd_utc = h->vis_time_start_mjd_utc +
            (h->vis_time_range[0] * h->time_inc_sec * SEC2DAYS);
    if (h->channel_snapshots)
    {
        h->im_freq_start_hz = h->vis_freq_start_hz +
                h->vis_chan_range[0] * h->freq_inc_hz;
    }
    else
    {
        double chan0 = 0.5 * (h->vis_chan_range[1] - h->vis_chan_range[0]);
        h->im_freq_start_hz = h->vis_freq_start_hz + chan0 * h->freq_inc_hz;
    }
    if (*status) return;

    if (h->vis_num_pols == 1 && !(h->im_type == OSKAR_IMAGE_TYPE_STOKES_I ||
            h->im_type == OSKAR_IMAGE_TYPE_PSF))
    {
        oskar_log_error(log, "Incompatible image type selected for scalar "
                "(Stokes-I) visibility data.");
        *status = OSKAR_ERR_SETTINGS_IMAGE;
        return;
    }

    /* Create the image and size it. */
    h->im = oskar_image_create(h->prec, OSKAR_CPU, status);
    oskar_image_resize(h->im, h->size, h->size,
            h->im_num_pols, h->im_num_times, h->im_num_chan, status);
    if (*status) return;
    oskar_image_set_fov(h->im, h->fov_deg, h->fov_deg);
    oskar_image_set_centre(h->im, h->im_ra_deg, h->im_dec_deg);
    oskar_image_set_time(h->im, h->im_time_start_mjd_utc, h->time_inc_sec);
    oskar_image_set_freq(h->im, h->im_freq_start_hz, h->freq_inc_hz);

    if (h->time_snapshots && h->channel_snapshots)
        h->num_vis = h->num_baselines;
    else if (h->time_snapshots && !h->channel_snapshots)
        h->num_vis = h->num_baselines * h->vis_num_channels;
    else if (!h->time_snapshots && h->channel_snapshots)
        h->num_vis = h->num_baselines * h->vis_num_times;
    else /* Time and frequency synthesis. */
        h->num_vis = h->num_baselines * h->vis_num_channels * h->vis_num_times;
    h->uu_im = oskar_mem_create(h->prec, OSKAR_CPU, h->num_vis, status);
    h->vv_im = oskar_mem_create(h->prec, OSKAR_CPU, h->num_vis, status);
    h->ww_im = oskar_mem_create(h->prec, OSKAR_CPU, h->num_vis, status);
    h->vis_im = oskar_mem_create(h->prec | OSKAR_COMPLEX, OSKAR_CPU, h->num_vis, status);
    h->stokes = oskar_mem_create(h->prec, OSKAR_CPU, h->num_vis, status);
    if (h->direction_type == OSKAR_IMAGE_DIRECTION_RA_DEC)
    {
        h->uu_tmp = oskar_mem_create(h->prec, OSKAR_CPU, h->num_vis, status);
        h->vv_tmp = oskar_mem_create(h->prec, OSKAR_CPU, h->num_vis, status);
        h->ww_tmp = oskar_mem_create(h->prec, OSKAR_CPU, h->num_vis, status);
    }

    /* Calculate pixel coordinate grid required for the DFT imager. */
    h->l = oskar_mem_create(h->prec, OSKAR_CPU, h->num_pixels, status);
    h->m = oskar_mem_create(h->prec, OSKAR_CPU, h->num_pixels, status);
    h->n = oskar_mem_create(h->prec, OSKAR_CPU, h->num_pixels, status);
    oskar_evaluate_image_lmn_grid(h->size, h->size, h->fov_rad, h->fov_rad,
            h->l, h->m, h->n, status);

    /* Pointer to slice of the image cube. */
    h->im_slice = oskar_mem_create_alias(0, 0, 0, status);

    /* If imaging away from the beam direction, evaluate l0-l, m0-m, n0-n
     * for the new pointing centre as well as a set of baseline coordinates
     * corresponding to the user specified imaging direction. */
    if (h->direction_type == OSKAR_IMAGE_DIRECTION_RA_DEC)
    {
        double l1, m1, n1, ra_rad, dec_rad, ra0_rad, dec0_rad;
        int num_elements;

        ra_rad = h->im_ra_deg * DEG2RAD;
        dec_rad = h->im_dec_deg * DEG2RAD;
        ra0_rad = h->vis_ra_deg * DEG2RAD;
        dec0_rad = h->vis_dec_deg * DEG2RAD;
        num_elements = h->num_baselines * h->vis_num_times;

        oskar_convert_lon_lat_to_relative_directions_d(1,
                &ra_rad, &dec_rad, ra0_rad, dec0_rad, &l1, &m1, &n1);
        h->delta_l = 0 - l1;
        h->delta_m = 0 - m1;
        h->delta_n = 1 - n1;

        h->uu_rot = oskar_mem_create(h->prec, OSKAR_CPU, num_elements, status);
        h->vv_rot = oskar_mem_create(h->prec, OSKAR_CPU, num_elements, status);
        h->ww_rot = oskar_mem_create(h->prec, OSKAR_CPU, num_elements, status);

        /* Work array for baseline evaluation. */
        h->work_uvw = oskar_mem_create(h->prec, OSKAR_CPU, 3 * h->num_stations, status);
        oskar_convert_ecef_to_baseline_uvw(h->num_stations,
                h->st_ecef_x, h->st_ecef_y, h->st_ecef_z, ra_rad, dec_rad,
                h->vis_num_times, h->vis_time_start_mjd_utc,
                h->time_inc_sec * SEC2DAYS, 0, h->uu_rot, h->vv_rot, h->ww_rot,
                h->work_uvw, status);
    }
}


void set_up_device_data(DeviceData* d, const HostData* h, int* status)
{
    d->l = oskar_mem_create_copy(h->l, OSKAR_GPU, status);
    d->m = oskar_mem_create_copy(h->m, OSKAR_GPU, status);
    d->n = oskar_mem_create_copy(h->n, OSKAR_GPU, status);
    d->uu = oskar_mem_create(h->prec, OSKAR_GPU, 0, status);
    d->vv = oskar_mem_create(h->prec, OSKAR_GPU, 0, status);
    d->ww = oskar_mem_create(h->prec, OSKAR_GPU, 0, status);
    d->amp = oskar_mem_create(h->prec | OSKAR_COMPLEX, OSKAR_GPU, 0, status);
    d->slice_gpu = oskar_mem_create(h->prec, OSKAR_GPU, h->num_pixels, status);
    d->slice_cpu = oskar_mem_create(h->prec, OSKAR_CPU, h->num_pixels, status);
    cudaDeviceSynchronize();
}


void free_host_data(HostData* h, int* status)
{
    oskar_mem_free(h->uu_rot, status);
    oskar_mem_free(h->vv_rot, status);
    oskar_mem_free(h->ww_rot, status);
    oskar_mem_free(h->work_uvw, status);
    oskar_mem_free(h->uu_tmp, status);
    oskar_mem_free(h->vv_tmp, status);
    oskar_mem_free(h->ww_tmp, status);
    oskar_mem_free(h->uu_im, status);
    oskar_mem_free(h->vv_im, status);
    oskar_mem_free(h->ww_im, status);
    oskar_mem_free(h->vis_im, status);
    oskar_mem_free(h->l, status);
    oskar_mem_free(h->m, status);
    oskar_mem_free(h->n, status);
    oskar_mem_free(h->im_slice, status);
    oskar_mem_free(h->stokes, status);
    oskar_binary_free(h->vis_file);
    oskar_vis_free(h->vis, status);
    oskar_vis_header_free(h->hdr, status);
    oskar_vis_block_free(h->blk, status);
    oskar_image_free(h->im, status);
    oskar_settings_old_free(&h->s);
    free(h);
}


void free_device_data(int num_gpus, int* cuda_device_ids,
        DeviceData* d, int* status)
{
    int i;
    if (!d) return;
    for (i = 0; i < num_gpus; ++i)
    {
        DeviceData* dd = &d[i];
        if (!dd) continue;
        cudaSetDevice(cuda_device_ids[i]);
        oskar_mem_free(dd->l, status);
        oskar_mem_free(dd->m, status);
        oskar_mem_free(dd->n, status);
        oskar_mem_free(dd->uu, status);
        oskar_mem_free(dd->vv, status);
        oskar_mem_free(dd->ww, status);
        oskar_mem_free(dd->amp, status);
        oskar_mem_free(dd->slice_gpu, status);
        oskar_mem_free(dd->slice_cpu, status);
        cudaDeviceReset();
    }
    free(d);
}


fitsfile* create_fits_file(const char* filename, int precision,
        int width, int height, int num_times, int num_channels,
        double centre_deg[2], double fov_deg[2], double start_time_mjd,
        double delta_time_sec, double start_freq_hz, double delta_freq_hz,
        const char* settings_log, size_t settings_log_length, int* status)
{
    int imagetype;
    long naxes[4];
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
    fits_create_img(f, imagetype, 4, naxes, status);
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


void get_baseline_coords(int num_baselines, const oskar_Mem* vis_uu,
        const oskar_Mem* vis_vv, const oskar_Mem* vis_ww,
        const int vis_time_range[2], const int vis_chan_range[2], int vis_time,
        double freq_start_hz, double freq_inc_hz, double im_freq,
        int time_snapshots, int channel_snapshots,
        oskar_Mem* uu, oskar_Mem* vv, oskar_Mem* ww, int* status)
{
    int c = 0, t = 0, i = 0;
    double freq, scaling;
    size_t offset1, offset2;
    oskar_Mem *uu_, *vv_, *ww_;

    /* Pointers into visibility coordinate arrays. */
    uu_ = oskar_mem_create_alias(0, 0, 0, status);
    vv_ = oskar_mem_create_alias(0, 0, 0, status);
    ww_ = oskar_mem_create_alias(0, 0, 0, status);

    /* Check whether time and/or frequency synthesis is being done. */
    if (time_snapshots && channel_snapshots)
    {
        offset2 = num_baselines * vis_time;
        oskar_mem_copy_contents(uu, vis_uu, 0, offset2, num_baselines, status);
        oskar_mem_copy_contents(vv, vis_vv, 0, offset2, num_baselines, status);
        oskar_mem_copy_contents(ww, vis_ww, 0, offset2, num_baselines, status);
    }
    else if (time_snapshots && !channel_snapshots) /* Freq synthesis */
    {
        for (c = vis_chan_range[0]; c <= vis_chan_range[1]; ++c, ++i)
        {
            /* Copy the baseline coordinates for the selected time. */
            offset1 = num_baselines * i; /* Destination */
            offset2 = num_baselines * vis_time; /* Source */
            oskar_mem_set_alias(uu_, uu, offset1, num_baselines, status);
            oskar_mem_set_alias(vv_, vv, offset1, num_baselines, status);
            oskar_mem_set_alias(ww_, ww, offset1, num_baselines, status);
            oskar_mem_copy_contents(uu_, vis_uu, 0, offset2,
                    num_baselines, status);
            oskar_mem_copy_contents(vv_, vis_vv, 0, offset2,
                    num_baselines, status);
            oskar_mem_copy_contents(ww_, vis_ww, 0, offset2,
                    num_baselines, status);

            /* Scale the coordinates with frequency. */
            freq = freq_start_hz + c * freq_inc_hz;
            scaling = freq / im_freq;
            oskar_mem_scale_real(uu_, scaling, status);
            oskar_mem_scale_real(vv_, scaling, status);
            oskar_mem_scale_real(ww_, scaling, status);
        }
    }
    else if (!time_snapshots && channel_snapshots) /* Time synthesis */
    {
        for (t = vis_time_range[0]; t <= vis_time_range[1]; ++t, ++i)
        {
            /* Copy the baseline coordinates for the current time. */
            offset1 = num_baselines * i; /* Destination */
            offset2 = num_baselines * t; /* Source */
            oskar_mem_copy_contents(uu, vis_uu, offset1, offset2,
                    num_baselines, status);
            oskar_mem_copy_contents(vv, vis_vv, offset1, offset2,
                    num_baselines, status);
            oskar_mem_copy_contents(ww, vis_ww, offset1, offset2,
                    num_baselines, status);
        }
    }
    else /* Time and frequency synthesis */
    {
        /* TODO These loops will need to be swapped. */
        for (c = vis_chan_range[0]; c <= vis_chan_range[1]; ++c)
        {
            freq = freq_start_hz + c * freq_inc_hz;
            scaling = freq / im_freq;

            for (t = vis_time_range[0]; t <= vis_time_range[1]; ++t, ++i)
            {
                /* Copy the baseline coordinates for the current time. */
                offset1 = num_baselines * i; /* Destination */
                offset2 = num_baselines * t; /* Source */
                oskar_mem_set_alias(uu_, uu, offset1, num_baselines, status);
                oskar_mem_set_alias(vv_, vv, offset1, num_baselines, status);
                oskar_mem_set_alias(ww_, ww, offset1, num_baselines, status);
                oskar_mem_copy_contents(uu_, vis_uu, 0, offset2,
                        num_baselines, status);
                oskar_mem_copy_contents(vv_, vis_vv, 0, offset2,
                        num_baselines, status);
                oskar_mem_copy_contents(ww_, vis_ww, 0, offset2,
                        num_baselines, status);

                /* Scale the coordinates with frequency. */
                oskar_mem_scale_real(uu_, scaling, status);
                oskar_mem_scale_real(vv_, scaling, status);
                oskar_mem_scale_real(ww_, scaling, status);
            }
        }
    }
    oskar_mem_free(uu_, status);
    oskar_mem_free(vv_, status);
    oskar_mem_free(ww_, status);
}


void get_stokes(const oskar_Mem* in, oskar_Mem* out, int* status)
{
    size_t i, num;
    if (*status) return;
    num = oskar_mem_length(in);
    if (!oskar_mem_is_matrix(in)) /* Already Stokes I. */
        oskar_mem_copy_contents(out, in, 0, 0, num, status);
    else
    {
        if (oskar_mem_precision(in) == OSKAR_DOUBLE)
        {
            const double4c* d_ = oskar_mem_double4c_const(in, status);
            double4c* s_ = oskar_mem_double4c(out, status);
            for (i = 0; i < num; ++i)
            {
                /* I = 0.5 (XX + YY) */
                s_[i].a.x =  0.5 * (d_[i].a.x + d_[i].d.x);
                s_[i].a.y =  0.5 * (d_[i].a.y + d_[i].d.y);
                /* Q = 0.5 (XX - YY) */
                s_[i].b.x =  0.5 * (d_[i].a.x - d_[i].d.x);
                s_[i].b.y =  0.5 * (d_[i].a.y - d_[i].d.y);
                /* U = 0.5 (XY + YX) */
                s_[i].c.x =  0.5 * (d_[i].b.x + d_[i].c.x);
                s_[i].c.y =  0.5 * (d_[i].b.y + d_[i].c.y);
                /* V = -0.5i (XY - YX) */
                s_[i].d.x =  0.5 * (d_[i].b.y - d_[i].c.y);
                s_[i].d.y = -0.5 * (d_[i].b.x - d_[i].c.x);
            }
        }
        else
        {
            const float4c* d_ = oskar_mem_float4c_const(in, status);
            float4c* s_ = oskar_mem_float4c(out, status);
            for (i = 0; i < num; ++i)
            {
                /* I = 0.5 (XX + YY) */
                s_[i].a.x =  0.5 * (d_[i].a.x + d_[i].d.x);
                s_[i].a.y =  0.5 * (d_[i].a.y + d_[i].d.y);
                /* Q = 0.5 (XX - YY) */
                s_[i].b.x =  0.5 * (d_[i].a.x - d_[i].d.x);
                s_[i].b.y =  0.5 * (d_[i].a.y - d_[i].d.y);
                /* U = 0.5 (XY + YX) */
                s_[i].c.x =  0.5 * (d_[i].b.x + d_[i].c.x);
                s_[i].c.y =  0.5 * (d_[i].b.y + d_[i].c.y);
                /* V = -0.5i (XY - YX) */
                s_[i].d.x =  0.5 * (d_[i].b.y - d_[i].c.y);
                s_[i].d.y = -0.5 * (d_[i].b.x - d_[i].c.x);
            }
        }
    }
}


void get_vis_amps(HostData* h, oskar_Mem* amps, const oskar_Mem* vis_in,
        int num_times, int vis_channel, int vis_time, int vis_pol, int* status)
{
    int b = 0, c = 0, i = 0, t = 0, idx = 0;
    int num_baselines, pol, pol_offset = 0, stride;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get pol_offset based on image type. */
    pol = h->im_type;
    if (pol == OSKAR_IMAGE_TYPE_STOKES_I || pol == OSKAR_IMAGE_TYPE_POL_XX)
        pol_offset = 0;
    if (pol == OSKAR_IMAGE_TYPE_STOKES_Q || pol == OSKAR_IMAGE_TYPE_POL_XY)
        pol_offset = 1;
    if (pol == OSKAR_IMAGE_TYPE_STOKES_U || pol == OSKAR_IMAGE_TYPE_POL_YX)
        pol_offset = 2;
    if (pol == OSKAR_IMAGE_TYPE_STOKES_V || pol == OSKAR_IMAGE_TYPE_POL_YY)
        pol_offset = 3;

    /* Override pol_offset if doing all polarisations. */
    if (pol == OSKAR_IMAGE_TYPE_STOKES || pol == OSKAR_IMAGE_TYPE_POL_LINEAR)
        pol_offset = vis_pol;

    /* Override pol_offset if scalar data. */
    if (!oskar_mem_is_matrix(vis_in)) pol_offset = 0;

    /* Set stride to 1 if visibilities are scalar, or 4 if polarised. */
    stride = oskar_mem_is_matrix(vis_in) ? 4 : 1;

    /* Copy the data out. */
    num_baselines = h->num_baselines;
    if (oskar_mem_precision(vis_in) == OSKAR_DOUBLE)
    {
        const double2* in;
        double2* out;
        in = oskar_mem_double2_const(vis_in, status);
        out = oskar_mem_double2(amps, status);

        if (h->time_snapshots && h->channel_snapshots)
        {
            idx = (vis_channel * num_times + vis_time) * num_baselines;
            for (b = 0; b < num_baselines; ++b)
                out[b] = in[stride * (idx + b) + pol_offset];
        }
        else if (h->time_snapshots && !h->channel_snapshots)
        { /* Frequency synthesis */
            for (c = h->vis_chan_range[0]; c <= h->vis_chan_range[1]; ++c)
            {
                idx = (c * num_times + vis_time) * num_baselines;
                for (b = 0; b < num_baselines; ++b, ++i)
                    out[i] = in[stride * (idx + b) + pol_offset];
            }
        }
        else if (!h->time_snapshots && h->channel_snapshots)
        { /* Time synthesis */
            for (t = h->vis_time_range[0]; t <= h->vis_time_range[1]; ++t)
            {
                idx = (vis_channel * num_times + t) * num_baselines;
                for (b = 0; b < num_baselines; ++b, ++i)
                    out[i] = in[stride * (idx + b) + pol_offset];
            }
        }
        else
        { /* Time and frequency synthesis */
            for (c = h->vis_chan_range[0]; c <= h->vis_chan_range[1]; ++c)
            {
                for (t = h->vis_time_range[0]; t <= h->vis_time_range[1]; ++t)
                {
                    idx = (c * num_times + t) * num_baselines;
                    for (b = 0; b < num_baselines; ++b, ++i)
                        out[i] = in[stride * (idx + b) + pol_offset];
                }
            }
        }
    }
    else /* Single precision */
    {
        const float2* in;
        float2* out;
        in = oskar_mem_float2_const(vis_in, status);
        out = oskar_mem_float2(amps, status);

        if (h->time_snapshots && h->channel_snapshots)
        {
            idx = (vis_channel * num_times + vis_time) * num_baselines;
            for (b = 0; b < num_baselines; ++b)
                out[b] = in[stride * (idx + b) + pol_offset];
        }
        else if (h->time_snapshots && !h->channel_snapshots)
        { /* Frequency synthesis */
            for (c = h->vis_chan_range[0]; c <= h->vis_chan_range[1]; ++c)
            {
                idx = (c * num_times + vis_time) * num_baselines;
                for (b = 0; b < num_baselines; ++b, ++i)
                    out[i] = in[stride * (idx + b) + pol_offset];
            }
        }
        else if (!h->time_snapshots && h->channel_snapshots)
        { /* Time synthesis */
            for (t = h->vis_time_range[0]; t <= h->vis_time_range[1]; ++t)
            {
                idx = (vis_channel * num_times + t) * num_baselines;
                for (b = 0; b < num_baselines; ++b, ++i)
                    out[i] = in[stride * (idx + b) + pol_offset];
            }
        }
        else
        { /* Time and frequency synthesis */
            for (c = h->vis_chan_range[0]; c <= h->vis_chan_range[1]; ++c)
            {
                for (t = h->vis_time_range[0]; t <= h->vis_time_range[1]; ++t)
                {
                    idx = (c * num_times + t) * num_baselines;
                    for (b = 0; b < num_baselines; ++b, ++i)
                        out[i] = in[stride * (idx + b) + pol_offset];
                }
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
