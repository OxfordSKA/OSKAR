/*
 * Copyright (c) 2016-2017, The University of Oxford
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

#include "imager/private_imager.h"

#include "convert/oskar_convert_cellsize_to_fov.h"
#include "convert/oskar_convert_fov_to_cellsize.h"
#include "convert/oskar_convert_lon_lat_to_relative_directions.h"
#include "imager/oskar_imager.h"
#include "imager/private_imager_composite_nearest_even.h"
#include "imager/private_imager_free_gpu_data.h"
#include "imager/private_imager_set_num_planes.h"
#include "math/oskar_cmath.h"
#include "utility/oskar_device_utils.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEG2RAD M_PI/180.0

#ifdef __cplusplus
extern "C" {
#endif

const char* oskar_imager_algorithm(const oskar_Imager* h)
{
    switch (h->algorithm)
    {
    case OSKAR_ALGORITHM_FFT:    return "FFT";
    case OSKAR_ALGORITHM_WPROJ:  return "W-projection";
    case OSKAR_ALGORITHM_DFT_2D: return "DFT 2D";
    case OSKAR_ALGORITHM_DFT_3D: return "DFT 3D";
    default:                     return "";
    }
}


double oskar_imager_cellsize(const oskar_Imager* h)
{
    return (h->cellsize_rad * (180.0 / M_PI)) * 3600.0;
}


int oskar_imager_channel_snapshots(const oskar_Imager* h)
{
    return h->chan_snaps;
}


int oskar_imager_coords_only(const oskar_Imager* h)
{
    return h->coords_only;
}


int oskar_imager_fft_on_gpu(const oskar_Imager* h)
{
    return h->fft_on_gpu;
}


double oskar_imager_fov(const oskar_Imager* h)
{
    return h->fov_deg;
}


double oskar_imager_freq_max_hz(const oskar_Imager* h)
{
    return h->freq_max_hz == 0.0 ? 0.0 : h->freq_max_hz - 0.01;
}


double oskar_imager_freq_min_hz(const oskar_Imager* h)
{
    return h->freq_min_hz == 0.0 ? 0.0 : h->freq_min_hz + 0.01;
}


int oskar_imager_generate_w_kernels_on_gpu(const oskar_Imager* h)
{
    return h->generate_w_kernels_on_gpu;
}


int oskar_imager_image_size(const oskar_Imager* h)
{
    return h->image_size;
}


const char* oskar_imager_image_type(const oskar_Imager* h)
{
    switch (h->im_type)
    {
    case OSKAR_IMAGE_TYPE_STOKES: return "Stokes";
    case OSKAR_IMAGE_TYPE_I:      return "I";
    case OSKAR_IMAGE_TYPE_Q:      return "Q";
    case OSKAR_IMAGE_TYPE_U:      return "U";
    case OSKAR_IMAGE_TYPE_V:      return "V";
    case OSKAR_IMAGE_TYPE_LINEAR: return "Linear";
    case OSKAR_IMAGE_TYPE_XX:     return "XX";
    case OSKAR_IMAGE_TYPE_XY:     return "XY";
    case OSKAR_IMAGE_TYPE_YX:     return "YX";
    case OSKAR_IMAGE_TYPE_YY:     return "YY";
    case OSKAR_IMAGE_TYPE_PSF:    return "PSF";
    default:                      return "";
    }
}


char* const* oskar_imager_input_files(const oskar_Imager* h)
{
    return h->input_files;
}


const char* oskar_imager_ms_column(const oskar_Imager* h)
{
    return h->ms_column;
}


int oskar_imager_num_image_planes(const oskar_Imager* h)
{
    return h->num_planes;
}


int oskar_imager_num_input_files(const oskar_Imager* h)
{
    return h->num_files;
}


int oskar_imager_num_w_planes(const oskar_Imager* h)
{
    return h->num_w_planes;
}


const char* oskar_imager_output_root(const oskar_Imager* h)
{
    return h->image_root;
}


int oskar_imager_plane_size(oskar_Imager* h)
{
    if (h->grid_size == 0)
    {
        if (h->algorithm == OSKAR_ALGORITHM_WPROJ)
        {
            (void) oskar_imager_composite_nearest_even(h->image_padding *
                    ((double)(h->image_size)) - 0.5, 0, &h->grid_size);
        }
        else
        {
            h->grid_size = h->image_size;
        }
    }
    return h->grid_size;
}


int oskar_imager_plane_type(const oskar_Imager* h)
{
    switch (h->algorithm)
    {
    case OSKAR_ALGORITHM_DFT_2D:
    case OSKAR_ALGORITHM_DFT_3D:
        return h->imager_prec;
    default:
        return h->imager_prec | OSKAR_COMPLEX;
    }
}


int oskar_imager_precision(const oskar_Imager* h)
{
    return h->imager_prec;
}


int oskar_imager_scale_norm_with_num_input_files(const oskar_Imager* h)
{
    return h->scale_norm_with_num_input_files;
}


void oskar_imager_set_algorithm(oskar_Imager* h, const char* type,
        int* status)
{
    if (*status) return;
    h->image_padding = 1.0;
    if (!strncmp(type, "FFT", 3) || !strncmp(type, "fft", 3))
    {
        h->algorithm = OSKAR_ALGORITHM_FFT;
        h->kernel_type = 'S';
        h->support = 3;
        h->oversample = 100;
    }
    else if (!strncmp(type, "W", 1) || !strncmp(type, "w", 1))
    {
        h->algorithm = OSKAR_ALGORITHM_WPROJ;
        h->oversample = 4;
        h->image_padding = 1.2;
    }
    else if (!strncmp(type, "DFT 2", 5) || !strncmp(type, "dft 2", 5))
        h->algorithm = OSKAR_ALGORITHM_DFT_2D;
    else if (!strncmp(type, "DFT 3", 5) || !strncmp(type, "dft 3", 5))
        h->algorithm = OSKAR_ALGORITHM_DFT_3D;
    else *status = OSKAR_ERR_INVALID_ARGUMENT;

    /* Recalculate grid plane size. */
    h->grid_size = 0;
    oskar_imager_reset_cache(h, status);
    (void) oskar_imager_plane_size(h);
}


void oskar_imager_set_cellsize(oskar_Imager* h, double cellsize_arcsec)
{
    h->set_cellsize = 1;
    h->set_fov = 0;
    h->cellsize_rad = (cellsize_arcsec / 3600.0) * (M_PI / 180.0);
    h->fov_deg = oskar_convert_cellsize_to_fov(
            h->cellsize_rad, h->image_size) * (180.0 / M_PI);
}


void oskar_imager_set_channel_snapshots(oskar_Imager* h, int value)
{
    h->chan_snaps = value;
}


void oskar_imager_set_coords_only(oskar_Imager* h, int flag)
{
    h->coords_only = flag;

    /* Check if coordinate input is starting or finishing. */
    if (flag)
    {
        /* Starting. */
        h->ww_min = DBL_MAX;
        h->ww_max = -DBL_MAX;
        h->ww_points = 0;
        h->ww_rms = 0.0;
    }
    else
    {
        /* Finishing. */
        if (h->ww_points > 0)
            h->ww_rms = sqrt(h->ww_rms / h->ww_points);

        /* Calculate required number of w-planes if not set. */
        if ((h->ww_max > 0.0) && (h->num_w_planes < 1))
        {
            double max_uvw, ww_mid;
            max_uvw = 1.05 * h->ww_max;
            ww_mid = 0.5 * (h->ww_min + h->ww_max);
            if (h->ww_rms > ww_mid)
                max_uvw *= h->ww_rms / ww_mid;
            h->num_w_planes = (int)(max_uvw *
                    fabs(sin(h->cellsize_rad * h->image_size / 2.0)));
        }
    }
}


void oskar_imager_set_default_direction(oskar_Imager* h)
{
    h->direction_type = 'O';
}


void oskar_imager_set_direction(oskar_Imager* h, double ra_deg, double dec_deg)
{
    h->direction_type = 'R';
    h->im_centre_deg[0] = ra_deg;
    h->im_centre_deg[1] = dec_deg;
}


void oskar_imager_set_fov(oskar_Imager* h, double fov_deg)
{
    h->set_cellsize = 0;
    h->set_fov = 1;
    h->fov_deg = fov_deg;
    h->cellsize_rad = oskar_convert_fov_to_cellsize(
            h->fov_deg * (M_PI / 180.0), h->image_size);
}


void oskar_imager_set_fft_on_gpu(oskar_Imager* h, int value)
{
    h->fft_on_gpu = value;
}


void oskar_imager_set_freq_max_hz(oskar_Imager* h, double max_freq_hz)
{
    if (max_freq_hz != 0.0 && max_freq_hz != DBL_MAX)
        max_freq_hz += 0.01;
    h->freq_max_hz = max_freq_hz;
}


void oskar_imager_set_freq_min_hz(oskar_Imager* h, double min_freq_hz)
{
    if (min_freq_hz != 0.0)
        min_freq_hz -= 0.01;
    h->freq_min_hz = min_freq_hz;
}


void oskar_imager_set_generate_w_kernels_on_gpu(oskar_Imager* h, int value)
{
    h->generate_w_kernels_on_gpu = value;
}


void oskar_imager_set_gpus(oskar_Imager* h, int num, const int* ids,
        int* status)
{
    int i, num_gpus_avail;
    if (*status) return;
    oskar_imager_free_gpu_data(h, status);
    num_gpus_avail = oskar_device_count(status);
    if (*status) return;
    if (num < 0)
    {
        h->num_gpus = num_gpus_avail;
        h->cuda_device_ids = (int*) calloc(h->num_gpus, sizeof(int));
        for (i = 0; i < h->num_gpus; ++i)
            h->cuda_device_ids[i] = i;
    }
    else if (num > 0)
    {
        if (num > num_gpus_avail)
        {
            *status = OSKAR_ERR_COMPUTE_DEVICES;
            return;
        }
        h->num_gpus = num;
        h->cuda_device_ids = (int*) calloc(h->num_gpus, sizeof(int));
        for (i = 0; i < h->num_gpus; ++i)
            h->cuda_device_ids[i] = ids[i];
    }
    else return;
    h->d = (DeviceData*) calloc(h->num_gpus, sizeof(DeviceData));
    for (i = 0; i < h->num_gpus; ++i)
    {
        oskar_device_set(h->cuda_device_ids[i], status);
        if (*status) return;
        h->d[i].uu = oskar_mem_create(h->imager_prec, OSKAR_GPU, 0, status);
        h->d[i].vv = oskar_mem_create(h->imager_prec, OSKAR_GPU, 0, status);
        h->d[i].ww = oskar_mem_create(h->imager_prec, OSKAR_GPU, 0, status);
        h->d[i].weight = oskar_mem_create(h->imager_prec, OSKAR_GPU, 0, status);
        h->d[i].amp = oskar_mem_create(h->imager_prec | OSKAR_COMPLEX,
                OSKAR_GPU, 0, status);
        h->d[i].l = oskar_mem_create(h->imager_prec, OSKAR_GPU, 0, status);
        h->d[i].m = oskar_mem_create(h->imager_prec, OSKAR_GPU, 0, status);
        h->d[i].n = oskar_mem_create(h->imager_prec, OSKAR_GPU, 0, status);
        h->d[i].block_gpu = oskar_mem_create(h->imager_prec,
                OSKAR_GPU, 0, status);
        h->d[i].block_cpu = oskar_mem_create(h->imager_prec,
                OSKAR_CPU, 0, status);
        h->d[i].plane_gpu = oskar_mem_create(h->imager_prec | OSKAR_COMPLEX,
                OSKAR_GPU, 0, status);
        oskar_device_synchronize();
    }
}


void oskar_imager_set_grid_kernel(oskar_Imager* h, const char* type,
        int support, int oversample, int* status)
{
    h->support = support;
    h->oversample = oversample;
    if (!strncmp(type, "S", 1) || !strncmp(type, "s", 1))
        h->kernel_type = 'S';
    else if (!strncmp(type, "G", 1) || !strncmp(type, "g", 1))
        h->kernel_type = 'G';
    else if (!strncmp(type, "P", 1) || !strncmp(type, "p", 1))
        h->kernel_type = 'P';
    else *status = OSKAR_ERR_INVALID_ARGUMENT;
}


void oskar_imager_set_image_size(oskar_Imager* h, int size, int* status)
{
    oskar_imager_set_size(h, size, status);
}


void oskar_imager_set_image_type(oskar_Imager* h, const char* type,
        int* status)
{
    if (*status) return;
    if (!strncmp(type, "S", 1) || !strncmp(type, "s", 1))
        h->im_type = OSKAR_IMAGE_TYPE_STOKES;
    else if (!strncmp(type, "I",  1) || !strncmp(type, "i",  1))
        h->im_type = OSKAR_IMAGE_TYPE_I;
    else if (!strncmp(type, "Q",  1) || !strncmp(type, "q",  1))
        h->im_type = OSKAR_IMAGE_TYPE_Q;
    else if (!strncmp(type, "U",  1) || !strncmp(type, "u",  1))
        h->im_type = OSKAR_IMAGE_TYPE_U;
    else if (!strncmp(type, "V",  1) || !strncmp(type, "v",  1))
        h->im_type = OSKAR_IMAGE_TYPE_V;
    else if (!strncmp(type, "P",  1) || !strncmp(type, "p",  1))
        h->im_type = OSKAR_IMAGE_TYPE_PSF;
    else if (!strncmp(type, "L",  1) || !strncmp(type, "l",  1))
        h->im_type = OSKAR_IMAGE_TYPE_LINEAR;
    else if (!strncmp(type, "XX", 2) || !strncmp(type, "xx", 2))
        h->im_type = OSKAR_IMAGE_TYPE_XX;
    else if (!strncmp(type, "XY", 2) || !strncmp(type, "xy", 2))
        h->im_type = OSKAR_IMAGE_TYPE_XY;
    else if (!strncmp(type, "YX", 2) || !strncmp(type, "yx", 2))
        h->im_type = OSKAR_IMAGE_TYPE_YX;
    else if (!strncmp(type, "YY", 2) || !strncmp(type, "yy", 2))
        h->im_type = OSKAR_IMAGE_TYPE_YY;
    else *status = OSKAR_ERR_INVALID_ARGUMENT;
    h->use_stokes = (h->im_type == OSKAR_IMAGE_TYPE_STOKES ||
            h->im_type == OSKAR_IMAGE_TYPE_I ||
            h->im_type == OSKAR_IMAGE_TYPE_Q ||
            h->im_type == OSKAR_IMAGE_TYPE_U ||
            h->im_type == OSKAR_IMAGE_TYPE_V);
    h->num_im_pols = (h->im_type == OSKAR_IMAGE_TYPE_STOKES ||
            h->im_type == OSKAR_IMAGE_TYPE_LINEAR) ? 4 : 1;
    if (h->im_type == OSKAR_IMAGE_TYPE_I || h->im_type == OSKAR_IMAGE_TYPE_XX)
        h->pol_offset = 0;
    if (h->im_type == OSKAR_IMAGE_TYPE_Q || h->im_type == OSKAR_IMAGE_TYPE_XY)
        h->pol_offset = 1;
    if (h->im_type == OSKAR_IMAGE_TYPE_U || h->im_type == OSKAR_IMAGE_TYPE_YX)
        h->pol_offset = 2;
    if (h->im_type == OSKAR_IMAGE_TYPE_V || h->im_type == OSKAR_IMAGE_TYPE_YY)
        h->pol_offset = 3;
}


void oskar_imager_set_input_files(oskar_Imager* h, int num_files,
        char* const* filenames, int* status)
{
    int i;
    if (*status) return;
    for (i = 0; i < h->num_files; ++i)
        free(h->input_files[i]);
    free(h->input_files);
    h->input_files = 0;
    h->num_files = num_files;
    if (num_files == 0) return;
    h->input_files = (char**) calloc(num_files, sizeof(char*));
    for (i = 0; i < num_files; ++i)
    {
        int len = 0;
        if (filenames[i]) len = strlen(filenames[i]);
        if (len > 0)
        {
            h->input_files[i] = (char*) calloc(1 + len, sizeof(char));
            strcpy(h->input_files[i], filenames[i]);
        }
    }
}


void oskar_imager_set_log(oskar_Imager* h, oskar_Log* log)
{
    h->log = log;
}


void oskar_imager_set_ms_column(oskar_Imager* h, const char* column,
        int* status)
{
    int len = 0;
    if (*status) return;
    len = strlen(column);
    if (len == 0) { *status = OSKAR_ERR_INVALID_ARGUMENT; return; }
    free(h->ms_column);
    h->ms_column = calloc(1 + len, 1);
    strcpy(h->ms_column, column);
}


void oskar_imager_set_output_root(oskar_Imager* h, const char* filename)
{
    int len = 0;
    free(h->image_root);
    h->image_root = 0;
    if (filename) len = strlen(filename);
    if (len > 0)
    {
        h->image_root = calloc(1 + len, 1);
        strcpy(h->image_root, filename);
    }
}


void oskar_imager_set_oversample(oskar_Imager* h, int value)
{
    h->oversample = value;
}


void oskar_imager_set_scale_norm_with_num_input_files(oskar_Imager* h,
        int value)
{
    h->scale_norm_with_num_input_files = value;
}


void oskar_imager_set_size(oskar_Imager* h, int size, int* status)
{
    if (size < 2 || size % 2 != 0)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    h->image_size = size;
    h->grid_size = 0;
    oskar_imager_reset_cache(h, status);
    (void) oskar_imager_plane_size(h);
    if (h->set_fov)
        h->cellsize_rad = oskar_convert_fov_to_cellsize(
                h->fov_deg * (M_PI / 180.0), h->image_size);
    else if (h->set_cellsize)
        h->fov_deg = oskar_convert_cellsize_to_fov(
                h->cellsize_rad, h->image_size) * (180.0 / M_PI);
}


void oskar_imager_set_time_max_utc(oskar_Imager* h, double time_max_mjd_utc)
{
    if (time_max_mjd_utc != 0.0 && time_max_mjd_utc != DBL_MAX)
        time_max_mjd_utc += 0.01 / 86400.0;
    h->time_max_utc = time_max_mjd_utc;
}


void oskar_imager_set_time_min_utc(oskar_Imager* h, double time_min_mjd_utc)
{
    if (time_min_mjd_utc != 0.0)
        time_min_mjd_utc -= 0.01 / 86400.0;
    h->time_min_utc = time_min_mjd_utc;
}


void oskar_imager_set_time_snapshots(oskar_Imager* h, int value)
{
    h->time_snaps = value;
}


void oskar_imager_set_uv_filter_max(oskar_Imager* h, double max_wavelength)
{
    h->uv_filter_max = max_wavelength;
}


void oskar_imager_set_uv_filter_min(oskar_Imager* h, double min_wavelength)
{
    h->uv_filter_min = min_wavelength;
}


static int qsort_compare_doubles(const void* a, const void* b)
{
    double aa, bb;
    aa = *(const double*)a;
    bb = *(const double*)b;
    if (aa < bb) return -1;
    if (aa > bb) return  1;
    return 0;
}


static void update_set(double ref, double inc, int num_to_check,
        int* num_recorded, double** values, double tol,
        double min_val, double max_val)
{
    int i, j;
    for (i = 0; i < num_to_check; ++i)
    {
        double value = ref + i * inc;
        for (j = 0; j < *num_recorded; ++j)
            if (fabs(value - (*values)[j]) < tol) break;
        if (j == *num_recorded &&
                value >= min_val && (value <= max_val || max_val <= 0.0))
        {
            (*num_recorded)++;
            *values = (double*) realloc(*values,
                    *num_recorded * sizeof(double));
            (*values)[j] = value;
        }
    }
    qsort(*values, *num_recorded, sizeof(double), qsort_compare_doubles);
}


void oskar_imager_set_vis_frequency(oskar_Imager* h,
        double ref_hz, double inc_hz, int num)
{
    h->vis_freq_start_hz = ref_hz;
    h->freq_inc_hz = inc_hz;
    if (!h->planes)
        update_set(ref_hz, inc_hz,
                num, &(h->num_sel_freqs), &(h->sel_freqs), 0.01,
                h->freq_min_hz, h->freq_max_hz);
}


void oskar_imager_set_vis_phase_centre(oskar_Imager* h,
        double ra_deg, double dec_deg)
{
    /* If imaging away from the beam direction, evaluate l0-l, m0-m, n0-n
     * for the new pointing centre, and a rotation matrix to generate the
     * rotated baseline coordinates. */
    if (h->direction_type == 'R')
    {
        double l1, m1, n1, ra_rad, dec_rad, ra0_rad, dec0_rad;
        double d_a, d_d, *M;

        ra_rad = h->im_centre_deg[0] * DEG2RAD;
        dec_rad = h->im_centre_deg[1] * DEG2RAD;
        ra0_rad = ra_deg * DEG2RAD;
        dec0_rad = dec_deg * DEG2RAD;
        d_a = ra0_rad - ra_rad; /* These are meant to be swapped: -delta_ra. */
        d_d = dec_rad - dec0_rad;

        /* Rotate by -delta_ra around v, then delta_dec around u. */
        M = h->M;
        M[0] = cos(d_a);           M[1] = 0.0;      M[2] = sin(d_a);
        M[3] = sin(d_a)*sin(d_d);  M[4] = cos(d_d); M[5] = -cos(d_a)*sin(d_d);
        M[6] = -sin(d_a)*cos(d_d); M[7] = sin(d_a); M[8] = cos(d_a)*cos(d_d);

        oskar_convert_lon_lat_to_relative_directions_d(1,
                &ra_rad, &dec_rad, ra0_rad, dec0_rad, &l1, &m1, &n1);
        h->delta_l = 0 - l1;
        h->delta_m = 0 - m1;
        h->delta_n = 1 - n1;
    }
    else
    {
        h->im_centre_deg[0] = ra_deg;
        h->im_centre_deg[1] = dec_deg;
    }
}


void oskar_imager_set_vis_time(oskar_Imager* h,
        double ref_mjd_utc, double inc_sec, int num)
{
    h->vis_time_start_mjd_utc = ref_mjd_utc;
    h->time_inc_sec = inc_sec;
    if (!h->planes)
        update_set(ref_mjd_utc + 0.5 * inc_sec / 86400.0, inc_sec / 86400.0,
                num, &(h->num_sel_times), &(h->sel_times), 0.01 / 86400.0,
                h->time_min_utc, h->time_max_utc);
}


void oskar_imager_set_num_w_planes(oskar_Imager* h, int value)
{
    h->num_w_planes = value;
}


void oskar_imager_set_weighting(oskar_Imager* h, const char* type, int* status)
{
    if (!strncmp(type, "N", 1) || !strncmp(type, "n", 1))
        h->weighting = OSKAR_WEIGHTING_NATURAL;
    else if (!strncmp(type, "R", 1) || !strncmp(type, "r", 1))
        h->weighting = OSKAR_WEIGHTING_RADIAL;
    else if (!strncmp(type, "U", 1) || !strncmp(type, "u", 1))
        h->weighting = OSKAR_WEIGHTING_UNIFORM;
    else *status = OSKAR_ERR_INVALID_ARGUMENT;
}


int oskar_imager_size(const oskar_Imager* h)
{
    return h->image_size;
}


double oskar_imager_time_max_utc(const oskar_Imager* h)
{
    return h->time_max_utc == 0.0 ? 0.0 : h->time_max_utc - 0.01 / 86400.0;
}


double oskar_imager_time_min_utc(const oskar_Imager* h)
{
    return h->time_min_utc == 0.0 ? 0.0 : h->time_min_utc + 0.01 / 86400.0;
}


int oskar_imager_time_snapshots(const oskar_Imager* h)
{
    return h->time_snaps;
}


double oskar_imager_uv_filter_max(const oskar_Imager* h)
{
    return h->uv_filter_max;
}


double oskar_imager_uv_filter_min(const oskar_Imager* h)
{
    return h->uv_filter_min;
}


const char* oskar_imager_weighting(const oskar_Imager* h)
{
    switch (h->weighting)
    {
    case OSKAR_WEIGHTING_NATURAL: return "Natural";
    case OSKAR_WEIGHTING_RADIAL:  return "Radial";
    case OSKAR_WEIGHTING_UNIFORM: return "Uniform";
    default:                      return "";
    }
}


#ifdef __cplusplus
}
#endif
