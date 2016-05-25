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

#include <cuda_runtime_api.h>
#include <private_imager.h>

#include <oskar_imager.h>
#include <private_imager_free_gpu_data.h>

#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static void oskar_imager_data_range(const int settings_range[2],
        int num_data_values, int range[2], int* status);


int oskar_imager_plane_size(oskar_Imager* h)
{
    return h->grid_size;
}


int oskar_imager_w_planes(oskar_Imager* h)
{
    return h->num_w_planes;
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


void oskar_imager_set_channel_range(oskar_Imager* h, int start, int end,
        int snapshots)
{
    h->chan_range[0] = start;
    h->chan_range[1] = end;
    h->chan_snaps = snapshots;
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
    h->fov_deg = fov_deg;
}


void oskar_imager_set_fft_on_gpu(oskar_Imager* h, int value)
{
    h->fft_on_gpu = value;
}


void oskar_imager_set_gpus(oskar_Imager* h, int num, const int* ids,
        int* status)
{
    int i, num_gpus_avail;
    if (*status) return;
    oskar_imager_free_gpu_data(h, status);
    *status = (int) cudaGetDeviceCount(&num_gpus_avail);
    if (*status) return;
    if (num > num_gpus_avail)
    {
        *status = OSKAR_ERR_CUDA_DEVICES;
        return;
    }
    if (num < 0)
    {
        h->num_gpus = num_gpus_avail;
        h->cuda_device_ids = (int*) calloc(h->num_gpus, sizeof(int));
        for (i = 0; i < h->num_gpus; ++i)
            h->cuda_device_ids[i] = i;
    }
    else if (num > 0)
    {
        h->num_gpus = num;
        h->cuda_device_ids = (int*) calloc(h->num_gpus, sizeof(int));
        for (i = 0; i < h->num_gpus; ++i)
            h->cuda_device_ids[i] = ids[i];
    }
    else return;
    h->d = (DeviceData*) calloc(h->num_gpus, sizeof(DeviceData));
    for (i = 0; i < h->num_gpus; ++i)
    {
        *status = (int) cudaSetDevice(h->cuda_device_ids[i]);
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
        cudaDeviceSynchronize();
    }
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


void oskar_imager_set_log(oskar_Imager* h, oskar_Log* log)
{
    h->log = log;
}


void oskar_imager_set_ms_column(oskar_Imager* h, const char* column,
        int* status)
{
    int len;
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
    int len;
    if (*status) return;
    len = strlen(filename);
    if (len == 0) { *status = OSKAR_ERR_FILE_IO; return; }
    free(h->image_root);
    h->image_root = calloc(1 + len, 1);
    strcpy(h->image_root, filename);
}


void oskar_imager_set_oversample(oskar_Imager* h, int value)
{
    h->oversample = value;
}


void oskar_imager_set_size(oskar_Imager* h, int size)
{
    h->image_size = size;
    h->grid_size = size;
}


void oskar_imager_set_time_range(oskar_Imager* h, int start, int end,
        int snapshots)
{
    h->time_range[0] = start;
    h->time_range[1] = end;
    h->time_snaps = snapshots;
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


void oskar_imager_set_w_planes(oskar_Imager* h, int value)
{
    h->num_w_planes = value;
}


void oskar_imager_set_w_range(oskar_Imager* h,
        double w_min, double w_max, double w_rms)
{
    h->ww_min = w_min;
    h->ww_max = w_max;
    h->ww_rms = w_rms;
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
