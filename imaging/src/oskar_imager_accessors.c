/*
 * Copyright (c) 2016, The University of Oxford
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

#include <private_imager.h>
#include <oskar_imager.h>

#include <oskar_convert_cellsize_to_fov.h>
#include <oskar_convert_fov_to_cellsize.h>
#include <oskar_device_utils.h>
#include <private_imager_free_gpu_data.h>
#include <private_imager_set_num_planes.h>

#include <oskar_cmath.h>
#include <float.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static void oskar_imager_data_range(const int settings_range[2],
        int num_data_values, int range[2], int* status);


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


int oskar_imager_channel_end(const oskar_Imager* h)
{
    return h->chan_range[1];
}


int oskar_imager_channel_snapshots(const oskar_Imager* h)
{
    return h->chan_snaps;
}


int oskar_imager_channel_start(const oskar_Imager* h)
{
    return h->chan_range[0];
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


const char* oskar_imager_input_file(const oskar_Imager* h)
{
    return h->input_file;
}


const char* oskar_imager_ms_column(const oskar_Imager* h)
{
    return h->ms_column;
}


int oskar_imager_num_image_planes(const oskar_Imager* h)
{
    return h->num_planes;
}


int oskar_imager_num_w_planes(const oskar_Imager* h)
{
    return h->num_w_planes;
}


const char* oskar_imager_output_root(const oskar_Imager* h)
{
    return h->image_root;
}


int oskar_imager_plane_size(const oskar_Imager* h)
{
    return h->grid_size;
}


int oskar_imager_plane_type(const oskar_Imager* h)
{
    if (h->num_planes <= 0 || !h->planes) return 0;
    return oskar_mem_type(h->planes[0]);
}


int oskar_imager_precision(const oskar_Imager* h)
{
    return h->imager_prec;
}


void oskar_imager_set_algorithm(oskar_Imager* h, const char* type,
        int* status)
{
    if (*status) return;
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
    }
    else if (!strncmp(type, "DFT 2", 5) || !strncmp(type, "dft 2", 5))
        h->algorithm = OSKAR_ALGORITHM_DFT_2D;
    else if (!strncmp(type, "DFT 3", 5) || !strncmp(type, "dft 3", 5))
        h->algorithm = OSKAR_ALGORITHM_DFT_3D;
    else *status = OSKAR_ERR_SETTINGS_IMAGE;
}


void oskar_imager_set_cellsize(oskar_Imager* h, double cellsize_arcsec)
{
    h->set_cellsize = 1;
    h->set_fov = 0;
    h->cellsize_rad = (cellsize_arcsec / 3600.0) * (M_PI / 180.0);
    h->fov_deg = oskar_convert_cellsize_to_fov(
            h->cellsize_rad, h->image_size) * (180.0 / M_PI);
}


void oskar_imager_set_channel_end(oskar_Imager* h, int value)
{
    h->chan_range[1] = value;
}


void oskar_imager_set_channel_snapshots(oskar_Imager* h, int value)
{
    h->chan_snaps = value;
}


void oskar_imager_set_channel_start(oskar_Imager* h, int value)
{
    h->chan_range[0] = value;
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
            *status = OSKAR_ERR_CUDA_DEVICES;
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
    else *status = OSKAR_ERR_SETTINGS_IMAGE;
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
    else *status = OSKAR_ERR_SETTINGS_IMAGE;
    h->use_stokes = (h->im_type == OSKAR_IMAGE_TYPE_STOKES ||
            h->im_type == OSKAR_IMAGE_TYPE_I ||
            h->im_type == OSKAR_IMAGE_TYPE_Q ||
            h->im_type == OSKAR_IMAGE_TYPE_U ||
            h->im_type == OSKAR_IMAGE_TYPE_V);
    h->im_num_pols = (h->im_type == OSKAR_IMAGE_TYPE_STOKES ||
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


void oskar_imager_set_input_file(oskar_Imager* h, const char* filename,
        int* status)
{
    int len = 0;
    if (*status) return;
    free(h->input_file);
    h->input_file = 0;
    if (filename) len = strlen(filename);
    if (len > 0)
    {
        h->input_file = calloc(1 + len, 1);
        strcpy(h->input_file, filename);
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
    if (len == 0) { *status = OSKAR_ERR_FILE_IO; return; }
    free(h->ms_column);
    h->ms_column = calloc(1 + len, 1);
    strcpy(h->ms_column, column);
}


void oskar_imager_set_output_root(oskar_Imager* h, const char* filename,
        int* status)
{
    int len = 0;
    if (*status) return;
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


void oskar_imager_set_size(oskar_Imager* h, int size, int* status)
{
    if (size < 2 || size % 2 != 0)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    h->image_size = size;
    h->grid_size = size;
    if (h->set_fov)
        h->cellsize_rad = oskar_convert_fov_to_cellsize(
                h->fov_deg * (M_PI / 180.0), h->image_size);
    else if (h->set_cellsize)
        h->fov_deg = oskar_convert_cellsize_to_fov(
                h->cellsize_rad, h->image_size) * (180.0 / M_PI);
}


void oskar_imager_set_time_end(oskar_Imager* h, int value)
{
    h->time_range[1] = value;
}


void oskar_imager_set_time_snapshots(oskar_Imager* h, int value)
{
    h->time_snaps = value;
}


void oskar_imager_set_time_start(oskar_Imager* h, int value)
{
    h->time_range[0] = value;
}


void oskar_imager_set_vis_frequency(oskar_Imager* h,
        double ref_hz, double inc_hz, int num, int* status)
{
    h->vis_freq_start_hz = ref_hz;
    h->freq_inc_hz = inc_hz;
    oskar_imager_data_range(h->chan_range, num,
            h->vis_chan_range, status);
}


void oskar_imager_set_vis_phase_centre(oskar_Imager* h,
        double ra_deg, double dec_deg)
{
    h->vis_centre_deg[0] = ra_deg;
    h->vis_centre_deg[1] = dec_deg;
}


void oskar_imager_set_vis_time(oskar_Imager* h,
        double ref_mjd_utc, double inc_sec, int num, int* status)
{
    h->vis_time_start_mjd_utc = ref_mjd_utc;
    h->time_inc_sec = inc_sec;
    oskar_imager_data_range(h->time_range, num,
            h->vis_time_range, status);
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
    else *status = OSKAR_ERR_SETTINGS_IMAGE;
}


int oskar_imager_size(const oskar_Imager* h)
{
    return h->image_size;;
}


int oskar_imager_time_end(const oskar_Imager* h)
{
    return h->time_range[1];
}


int oskar_imager_time_snapshots(const oskar_Imager* h)
{
    return h->time_snaps;
}


int oskar_imager_time_start(const oskar_Imager* h)
{
    return h->time_range[0];
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


void oskar_imager_data_range(const int settings_range[2],
        int num_data_values, int range[2], int* status)
{
    if (*status) return;
    if (settings_range[0] >= num_data_values ||
            settings_range[1] >= num_data_values)
    {
        *status = OSKAR_ERR_INVALID_RANGE;
        return;
    }
    range[0] = settings_range[0] < 0 ? 0 : settings_range[0];
    range[1] = settings_range[1] < 0 ? num_data_values - 1 : settings_range[1];
    if (range[0] > range[1])
    {
        *status = OSKAR_ERR_INVALID_RANGE;
        return;
    }
}


#ifdef __cplusplus
}
#endif
