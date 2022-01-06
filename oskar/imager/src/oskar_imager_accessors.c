/*
 * Copyright (c) 2016-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager.h"

#include "convert/oskar_convert_cellsize_to_fov.h"
#include "convert/oskar_convert_fov_to_cellsize.h"
#include "imager/oskar_imager.h"
#include "imager/private_imager_composite_nearest_even.h"
#include "imager/private_imager_free_device_data.h"
#include "imager/private_imager_set_num_planes.h"
#include "math/oskar_cmath.h"
#include "utility/oskar_device.h"
#include "utility/oskar_get_num_procs.h"

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


int oskar_imager_grid_on_gpu(const oskar_Imager* h)
{
    return h->grid_on_gpu;
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


oskar_Log* oskar_imager_log(oskar_Imager* h)
{
    return h->log;
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
    return h->output_root;
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
    if (*status || !type) return;
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
    {
        h->algorithm = OSKAR_ALGORITHM_DFT_2D;
    }
    else if (!strncmp(type, "DFT 3", 5) || !strncmp(type, "dft 3", 5))
    {
        h->algorithm = OSKAR_ALGORITHM_DFT_3D;
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
    }

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
        {
            h->ww_rms = sqrt(h->ww_rms / h->ww_points);
        }

        /* Calculate required number of w-planes if not set. */
        if ((h->ww_max > 0.0) && (h->num_w_planes < 1))
        {
            double max_uvw = 0.0, ww_mid = 0.0;
            max_uvw = 1.05 * h->ww_max;
            ww_mid = 0.5 * (h->ww_min + h->ww_max);
            if (h->ww_rms > ww_mid)
            {
                max_uvw *= h->ww_rms / ww_mid;
            }
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
    {
        max_freq_hz += 0.01;
    }
    h->freq_max_hz = max_freq_hz;
}


void oskar_imager_set_freq_min_hz(oskar_Imager* h, double min_freq_hz)
{
    if (min_freq_hz != 0.0)
    {
        min_freq_hz -= 0.01;
    }
    h->freq_min_hz = min_freq_hz;
}


void oskar_imager_set_generate_w_kernels_on_gpu(oskar_Imager* h, int value)
{
    h->generate_w_kernels_on_gpu = value;
}


void oskar_imager_set_gpus(oskar_Imager* h, int num, const int* ids,
        int* status)
{
    int i = 0;
    if (*status) return;
    oskar_imager_free_device_data(h, status);
    if (*status) return;
    if (num < 0)
    {
        free(h->gpu_ids);
        h->gpu_ids = (int*) calloc(h->num_gpus_avail, sizeof(int));
        h->num_gpus = 0;
        if (h->gpu_ids)
        {
            h->num_gpus = h->num_gpus_avail;
            for (i = 0; i < h->num_gpus; ++i) h->gpu_ids[i] = i;
        }
    }
    else if (num > 0)
    {
        if (num > h->num_gpus_avail)
        {
            oskar_log_error(h->log, "More GPUs were requested than found.");
            *status = OSKAR_ERR_COMPUTE_DEVICES;
            return;
        }
        free(h->gpu_ids);
        h->gpu_ids = (int*) calloc(num, sizeof(int));
        h->num_gpus = 0;
        if (h->gpu_ids)
        {
            h->num_gpus = num;
            for (i = 0; i < h->num_gpus; ++i) h->gpu_ids[i] = ids[i];
        }
    }
    else /* num == 0 */
    {
        free(h->gpu_ids);
        h->gpu_ids = 0;
        h->num_gpus = 0;
    }
    for (i = 0; (i < h->num_gpus) && h->gpu_ids; ++i)
    {
        oskar_device_set(h->dev_loc, h->gpu_ids[i], status);
        if (*status) return;
    }
}


void oskar_imager_set_grid_kernel(oskar_Imager* h, const char* type,
        int support, int oversample, int* status)
{
    if (*status || !type) return;
    h->support = support;
    h->oversample = oversample;
    if (!strncmp(type, "S", 1) || !strncmp(type, "s", 1))
    {
        h->kernel_type = 'S';
    }
    else if (!strncmp(type, "G", 1) || !strncmp(type, "g", 1))
    {
        h->kernel_type = 'G';
    }
    else if (!strncmp(type, "P", 1) || !strncmp(type, "p", 1))
    {
        h->kernel_type = 'P';
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
    }
}


void oskar_imager_set_grid_on_gpu(oskar_Imager* h, int value)
{
    h->grid_on_gpu = value;
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
    {
        h->im_type = OSKAR_IMAGE_TYPE_STOKES;
    }
    else if (!strncmp(type, "I",  1) || !strncmp(type, "i",  1))
    {
        h->im_type = OSKAR_IMAGE_TYPE_I;
    }
    else if (!strncmp(type, "Q",  1) || !strncmp(type, "q",  1))
    {
        h->im_type = OSKAR_IMAGE_TYPE_Q;
    }
    else if (!strncmp(type, "U",  1) || !strncmp(type, "u",  1))
    {
        h->im_type = OSKAR_IMAGE_TYPE_U;
    }
    else if (!strncmp(type, "V",  1) || !strncmp(type, "v",  1))
    {
        h->im_type = OSKAR_IMAGE_TYPE_V;
    }
    else if (!strncmp(type, "P",  1) || !strncmp(type, "p",  1))
    {
        h->im_type = OSKAR_IMAGE_TYPE_PSF;
    }
    else if (!strncmp(type, "L",  1) || !strncmp(type, "l",  1))
    {
        h->im_type = OSKAR_IMAGE_TYPE_LINEAR;
    }
    else if (!strncmp(type, "XX", 2) || !strncmp(type, "xx", 2))
    {
        h->im_type = OSKAR_IMAGE_TYPE_XX;
    }
    else if (!strncmp(type, "XY", 2) || !strncmp(type, "xy", 2))
    {
        h->im_type = OSKAR_IMAGE_TYPE_XY;
    }
    else if (!strncmp(type, "YX", 2) || !strncmp(type, "yx", 2))
    {
        h->im_type = OSKAR_IMAGE_TYPE_YX;
    }
    else if (!strncmp(type, "YY", 2) || !strncmp(type, "yy", 2))
    {
        h->im_type = OSKAR_IMAGE_TYPE_YY;
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
    }
    h->use_stokes = (h->im_type == OSKAR_IMAGE_TYPE_STOKES ||
            h->im_type == OSKAR_IMAGE_TYPE_I ||
            h->im_type == OSKAR_IMAGE_TYPE_Q ||
            h->im_type == OSKAR_IMAGE_TYPE_U ||
            h->im_type == OSKAR_IMAGE_TYPE_V);
    h->num_im_pols = (h->im_type == OSKAR_IMAGE_TYPE_STOKES ||
            h->im_type == OSKAR_IMAGE_TYPE_LINEAR) ? 4 : 1;
    if (h->im_type == OSKAR_IMAGE_TYPE_I || h->im_type == OSKAR_IMAGE_TYPE_XX)
    {
        h->pol_offset = 0;
    }
    if (h->im_type == OSKAR_IMAGE_TYPE_Q || h->im_type == OSKAR_IMAGE_TYPE_XY)
    {
        h->pol_offset = 1;
    }
    if (h->im_type == OSKAR_IMAGE_TYPE_U || h->im_type == OSKAR_IMAGE_TYPE_YX)
    {
        h->pol_offset = 2;
    }
    if (h->im_type == OSKAR_IMAGE_TYPE_V || h->im_type == OSKAR_IMAGE_TYPE_YY)
    {
        h->pol_offset = 3;
    }
}


void oskar_imager_set_input_files(oskar_Imager* h, int num_files,
        const char* const* filenames, int* status)
{
    int i = 0;
    if (*status) return;
    for (i = 0; i < h->num_files; ++i) free(h->input_files[i]);
    free(h->input_files);
    free(h->input_root);
    h->input_files = 0;
    h->input_root = 0;
    h->num_files = num_files;
    if (num_files == 0 || !filenames) return;
    h->input_files = (char**) calloc(num_files, sizeof(char*));
    if (!h->input_files) return;
    for (i = 0; i < num_files; ++i)
    {
        if (!filenames[i]) continue;
        const size_t len = strlen(filenames[i]);
        if (len == 0) continue;
        h->input_files[i] = (char*) calloc(1 + len, sizeof(char));
        if (h->input_files[i]) memcpy(h->input_files[i], filenames[i], len);
    }
    if (!filenames[0]) return;
    const size_t len = strlen(filenames[0]);
    h->input_root = (char*) calloc(1 + len, 1);
    if (h->input_root)
    {
        memcpy(h->input_root, filenames[0], len);
        char* ptr = strrchr(h->input_root, '.');
        if (ptr) *ptr = 0;
    }
    else
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
    }
}


void oskar_imager_set_ms_column(oskar_Imager* h, const char* column,
        int* status)
{
    if (*status || !column) return;
    const size_t len = strlen(column);
    if (len == 0) { *status = OSKAR_ERR_INVALID_ARGUMENT; return; }
    free(h->ms_column);
    h->ms_column = (char*) calloc(1 + len, 1);
    if (h->ms_column) memcpy(h->ms_column, column, len);
}


void oskar_imager_set_num_devices(oskar_Imager* h, int value)
{
    int status = 0;
    oskar_imager_free_device_data(h, &status);
    if (value < 1)
    {
        value = (h->num_gpus == 0) ? oskar_get_num_procs() : h->num_gpus;
    }
    if (value < 1) value = 1;
    h->num_devices = value;
    free(h->d);
    h->d = (DeviceData*) calloc(h->num_devices, sizeof(DeviceData));
}


void oskar_imager_set_output_root(oskar_Imager* h, const char* filename)
{
    size_t len = 0;
    free(h->output_root);
    h->output_root = 0;
    if (filename) len = strlen(filename);
    if (len > 0)
    {
        h->output_root = (char*) calloc(1 + len, 1);
        if (h->output_root) memcpy(h->output_root, filename, len);
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
    if (*status) return;
    if (size < 2 || size % 2 != 0)
    {
        oskar_log_error(h->log, "Need an even number of pixels "
                "for image side length.");
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    h->image_size = size;
    h->grid_size = 0;
    oskar_imager_reset_cache(h, status);
    (void) oskar_imager_plane_size(h);
    if (h->set_fov)
    {
        h->cellsize_rad = oskar_convert_fov_to_cellsize(
                h->fov_deg * (M_PI / 180.0), h->image_size);
    }
    else if (h->set_cellsize)
    {
        h->fov_deg = oskar_convert_cellsize_to_fov(
                h->cellsize_rad, h->image_size) * (180.0 / M_PI);
    }
}


void oskar_imager_set_time_max_utc(oskar_Imager* h, double time_max_mjd_utc)
{
    if (time_max_mjd_utc != 0.0 && time_max_mjd_utc != DBL_MAX)
    {
        time_max_mjd_utc += 0.01 / 86400.0;
    }
    h->time_max_utc = time_max_mjd_utc * 86400.0;
}


void oskar_imager_set_time_min_utc(oskar_Imager* h, double time_min_mjd_utc)
{
    if (time_min_mjd_utc != 0.0)
    {
        time_min_mjd_utc -= 0.01 / 86400.0;
    }
    h->time_min_utc = time_min_mjd_utc * 86400.0;
}


void oskar_imager_set_uv_filter_max(oskar_Imager* h, double max_wavelength)
{
    h->uv_filter_max = max_wavelength;
}


void oskar_imager_set_uv_filter_min(oskar_Imager* h, double min_wavelength)
{
    h->uv_filter_min = min_wavelength;
}


void oskar_imager_set_uv_taper(oskar_Imager* h,
        double taper_u_wavelength, double taper_v_wavelength)
{
    h->uv_taper[0] = taper_u_wavelength;
    h->uv_taper[1] = taper_v_wavelength;
}


static int qsort_compare_doubles(const void* a, const void* b)
{
    double aa = 0.0, bb = 0.0;
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
    int i = 0, j = 0;
    for (i = 0; i < num_to_check; ++i)
    {
        double value = ref + i * inc;
        for (j = 0; j < *num_recorded; ++j)
        {
            if (fabs(value - (*values)[j]) < tol) break;
        }
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
    {
        update_set(ref_hz, inc_hz, num,
                &(h->num_sel_freqs), &(h->sel_freqs), 0.01,
                h->freq_min_hz, h->freq_max_hz);
    }
}


void oskar_imager_set_vis_phase_centre(oskar_Imager* h,
        double ra_deg, double dec_deg)
{
    /* If imaging away from the beam direction, evaluate l0-l, m0-m, n0-n
     * for the new pointing centre, and a rotation matrix to generate the
     * rotated baseline coordinates. */
    if (h->direction_type == 'R')
    {
        double l1 = 0.0, m1 = 0.0, n1 = 0.0, d_a = 0.0, d_d = 0.0;
        double dec_rad = 0.0, dec0_rad = 0.0, *M = 0;
        double sin_d_a = 0.0, cos_d_a = 0.0, sin_d_d = 0.0, cos_d_d = 0.0;
        double sin_dec = 0.0, cos_dec = 0.0, sin_dec0 = 0.0, cos_dec0 = 0.0;

        /* Rotate by -delta_ra around v, then delta_dec around u. */
        dec_rad = h->im_centre_deg[1] * DEG2RAD;
        dec0_rad = dec_deg * DEG2RAD;
        d_a = (ra_deg - h->im_centre_deg[0]) * DEG2RAD; /* For -delta_ra. */
        d_d = (h->im_centre_deg[1] - dec_deg) * DEG2RAD;
        sin_d_a = sin(d_a);
        cos_d_a = cos(d_a);
        sin_d_d = sin(d_d);
        cos_d_d = cos(d_d);
        M = h->M;
        M[0] =  cos_d_a;           M[1] = 0.0;     M[2] =  sin_d_a;
        M[3] =  sin_d_a * sin_d_d; M[4] = cos_d_d; M[5] = -cos_d_a * sin_d_d;
        M[6] = -sin_d_a * cos_d_d; M[7] = sin_d_d; M[8] =  cos_d_a * cos_d_d;

        /* Convert from spherical to tangent-plane to get delta (l, m, n). */
        sin_dec0 = sin(dec0_rad);
        cos_dec0 = cos(dec0_rad);
        sin_dec  = sin(dec_rad);
        cos_dec  = cos(dec_rad);
        l1 = cos_dec  * -sin_d_a;
        m1 = cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_d_a;
        n1 = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_d_a;
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


void oskar_imager_set_num_w_planes(oskar_Imager* h, int value)
{
    h->num_w_planes = value;
    if (value > 0)
    {
        h->ww_max = 0.0;
        h->ww_min = 0.0;
        h->ww_points = 0;
        h->ww_rms = 0.0;
    }
}


void oskar_imager_set_weighting(oskar_Imager* h, const char* type, int* status)
{
    if (*status || !type) return;
    if (!strncmp(type, "N", 1) || !strncmp(type, "n", 1))
    {
        h->weighting = OSKAR_WEIGHTING_NATURAL;
    }
    else if (!strncmp(type, "R", 1) || !strncmp(type, "r", 1))
    {
        h->weighting = OSKAR_WEIGHTING_RADIAL;
    }
    else if (!strncmp(type, "U", 1) || !strncmp(type, "u", 1))
    {
        h->weighting = OSKAR_WEIGHTING_UNIFORM;
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
    }
}


int oskar_imager_size(const oskar_Imager* h)
{
    return h->image_size;
}


double oskar_imager_time_max_utc(const oskar_Imager* h)
{
    return h->time_max_utc == 0.0 ? 0.0 :
            (h->time_max_utc / 86400.0) - 0.01 / 86400.0;
}


double oskar_imager_time_min_utc(const oskar_Imager* h)
{
    return h->time_min_utc == 0.0 ? 0.0 :
            (h->time_min_utc / 86400.0) + 0.01 / 86400.0;
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
