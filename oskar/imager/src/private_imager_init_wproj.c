/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager.h"
#include "imager/oskar_imager.h"

#include "imager/oskar_grid_functions_spheroidal.h"
#include "imager/private_imager_composite_nearest_even.h"
#include "imager/private_imager_generate_w_phase_screen.h"
#include "imager/private_imager_init_wproj.h"
#include "math/oskar_cmath.h"
#include "math/oskar_fft.h"
#include "utility/oskar_device.h"
#include "utility/oskar_get_memory_usage.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#include <fitsio.h>

static void oskar_imager_evaluate_w_kernel_params(const oskar_Imager* h,
        int* num_w_planes, double* w_scale);

static oskar_Mem* oskar_imager_evaluate_w_kernel_cube(oskar_Imager* h,
        int num_w_planes, double w_scale,
        size_t* conv_size_half, double* norm_factor, int* status);

static oskar_Mem* oskar_imager_evaluate_w_kernel_support_sizes(
        int num_w_planes, int oversample, size_t conv_size_half,
        const oskar_Mem* kernel_cube, double norm_factor, int* status);

static void oskar_imager_normalise_kernel_cube(const oskar_Mem* support,
        int oversample, size_t conv_size_half, oskar_Mem* kernel_cube,
        int* status);

static void oskar_imager_trim_and_save_kernel_cube(oskar_Imager* h,
        int num_planes, oskar_Mem* support, size_t* conv_size_half,
        oskar_Mem* kernel_cube, int* status);

static void oskar_imager_rearrange_kernels(int num_w_planes,
        const oskar_Mem* support, int oversample, size_t conv_size_half,
        const oskar_Mem* kernels_in, oskar_Mem* kernels_out,
        int* rearranged_kernel_start, int* status);

/*
 * W-kernel generation is based on CASA implementation
 * in code/synthesis/TransformMachines/WPConvFunc.cc
 */
void oskar_imager_init_wproj(oskar_Imager* h, int* status)
{
    size_t conv_size_half = 0;
    double norm_factor = 1.;
    oskar_Mem *kernel_cube = 0;
    const int save_kernels = 0;
    if (*status) return;

    /* Evaluate number of w-projection planes, and w-scale. */
    oskar_imager_evaluate_w_kernel_params(h, &h->num_w_planes, &h->w_scale);

    /* Evaluate unnormalised kernels. */
    kernel_cube = oskar_imager_evaluate_w_kernel_cube(h, h->num_w_planes,
            h->w_scale, &conv_size_half, &norm_factor, status);

    /* Evaluate the support size of each kernel. */
    oskar_mem_free(h->w_support, status);
    h->w_support = oskar_imager_evaluate_w_kernel_support_sizes(
            h->num_w_planes, h->oversample, conv_size_half,
            kernel_cube, norm_factor, status);

#if 0
    /* Print kernel support sizes. */
    {
        int i = 0;
        for (i = 0; i < h->num_w_planes; ++i)
        {
            const int* supp = oskar_mem_int_const(h->w_support, status);
            printf("Plane %d, support: %d\n", i, supp[i]);
        }
    }
#endif

    /* Normalise the kernel cube. */
    oskar_imager_normalise_kernel_cube(h->w_support, h->oversample,
            conv_size_half, kernel_cube, status);
    if (save_kernels)
    {
        oskar_imager_trim_and_save_kernel_cube(h, h->num_w_planes,
                h->w_support, &conv_size_half, kernel_cube, status);
    }

    /* Rearrange and compact the kernels. */
    oskar_mem_free(h->w_kernels_compact, status);
    oskar_mem_free(h->w_kernel_start, status);
    h->w_kernel_start = oskar_mem_create(OSKAR_INT, OSKAR_CPU,
            h->num_w_planes, status);
    h->w_kernels_compact = oskar_mem_create(h->imager_prec| OSKAR_COMPLEX,
            OSKAR_CPU, 0, status);
    oskar_imager_rearrange_kernels(h->num_w_planes, h->w_support,
            h->oversample, conv_size_half, kernel_cube, h->w_kernels_compact,
            oskar_mem_int(h->w_kernel_start, status), status);
    oskar_mem_free(kernel_cube, status);

    /* Record data about the kernels. */
    oskar_log_message(h->log, 'M', 0, "Baseline W values (wavelengths)");
    oskar_log_message(h->log, 'M', 1, "Min: %.12e", h->ww_min);
    oskar_log_message(h->log, 'M', 1, "Max: %.12e", h->ww_max);
    oskar_log_message(h->log, 'M', 1, "RMS: %.12e", h->ww_rms);
    oskar_log_message(h->log, 'M', 0,
            "Using %d W-projection planes.", h->num_w_planes);
    oskar_log_message(h->log, 'M', 0,
            "Convolution kernel support range %d to %d.",
            oskar_mem_int(h->w_support, status)[0],
            oskar_mem_int(h->w_support, status)[h->num_w_planes - 1]);
    const size_t compacted_len = oskar_mem_length(h->w_kernels_compact) *
            oskar_mem_element_size(h->imager_prec | OSKAR_COMPLEX);
    oskar_log_message(h->log, 'M', 0, "Convolution kernels use %.1f MB.",
            compacted_len * 1e-6);

    /* Copy to device memory if required. */
    if (h->grid_on_gpu && h->num_gpus > 0)
    {
        int i = 0;
        if (h->num_devices < h->num_gpus)
        {
            oskar_imager_set_num_devices(h, h->num_gpus);
        }
        for (i = 0; i < h->num_gpus; ++i)
        {
            DeviceData* d = &h->d[i];
            oskar_device_set(h->dev_loc, h->gpu_ids[i], status);
            if (*status) break;
            oskar_mem_free(d->w_kernels_compact, status);
            oskar_mem_free(d->w_kernel_start, status);
            oskar_mem_free(d->w_support, status);
            d->w_kernels_compact = oskar_mem_create_copy(
                    h->w_kernels_compact, h->dev_loc, status);
            d->w_kernel_start = oskar_mem_create_copy(
                    h->w_kernel_start, h->dev_loc, status);
            d->w_support = oskar_mem_create_copy(
                    h->w_support, h->dev_loc, status);
        }

        /* No longer need kernels in host memory. */
        oskar_mem_free(h->w_kernels_compact, status);
        h->w_kernels_compact = 0;
    }
}


static void oskar_imager_evaluate_w_kernel_params(const oskar_Imager* h,
        int* num_w_planes, double* w_scale)
{
    double max_uvw = 0.0;
    if (h->ww_max > 0.0)
    {
        const double ww_mid = 0.5 * (h->ww_min + h->ww_max);
        max_uvw = 1.05 * h->ww_max;
        if (h->ww_rms > ww_mid)
        {
            max_uvw *= h->ww_rms / ww_mid;
        }
    }
    else
    {
        max_uvw = 0.25 / fabs(h->cellsize_rad);
    }
    if (*num_w_planes < 1)
    {
        *num_w_planes = (int)(max_uvw *
                fabs(sin(h->cellsize_rad * h->image_size / 2.0)));
    }
    if (*num_w_planes < 16) *num_w_planes = 16;
    *w_scale = pow(*num_w_planes - 1, 2.0) / max_uvw;
}


static oskar_Mem* oskar_imager_evaluate_w_kernel_cube(oskar_Imager* h,
        int num_w_planes, double w_scale,
        size_t* conv_size_half, double* norm_factor, int* status)
{
    size_t max_mem_bytes = 0;
    oskar_FFT* fft = 0;
    oskar_Mem *screen = 0, *screen_gpu = 0, *screen_ptr = 0;
    oskar_Mem *taper = 0, *taper_gpu = 0, *taper_ptr = 0;
    oskar_Mem *kernel_cube = 0;
    char *ptr_out = 0, *ptr_in = 0;
    double *maxes = 0, max_val = -INT_MAX, sampling = 0.0;
    int i = 0;
    if (*status) return 0;

    /* Calculate convolution kernel size. */
    const size_t max_bytes_per_plane = 64 * 1024 * 1024; /* 64 MB/plane */
    max_mem_bytes = oskar_get_total_physical_memory();
    max_mem_bytes = MIN(max_mem_bytes, max_bytes_per_plane * num_w_planes);
    const double max_conv_size = sqrt(max_mem_bytes / (16. * num_w_planes));
    const int nearest = oskar_imager_composite_nearest_even(
            2 * (int)(max_conv_size / 2.0), 0, 0);
    const int conv_size = MIN((int)(h->image_size * h->image_padding),nearest);
    *conv_size_half = conv_size / 2 - 1;
    const size_t kernel_plane_size = (*conv_size_half) * (*conv_size_half);

    /* Get size of inner region of kernel. */
    const int inner = conv_size / h->oversample;
    const double l_max = sin(0.5 * h->fov_deg * M_PI/180.0);
    sampling = (2.0 * l_max * h->oversample) / h->image_size;
    sampling *= ((double) oskar_imager_plane_size(h)) / ((double) conv_size);

    /* Generate 1D spheroidal tapering function to cover the inner region. */
    const int prec = h->imager_prec;
    taper = oskar_mem_create(prec, OSKAR_CPU, (size_t) inner, status);
    taper_ptr = taper;
    if (prec == OSKAR_DOUBLE)
    {
        double* t = (double*) oskar_mem_void(taper);
        for (i = 0; i < inner; ++i)
        {
            const double nu = (i - (inner / 2)) / ((double)(inner / 2));
            t[i] = oskar_grid_function_spheroidal(fabs(nu));
        }
    }
    else
    {
        float* t = (float*) oskar_mem_void(taper);
        for (i = 0; i < inner; ++i)
        {
            const double nu = (i - (inner / 2)) / ((double)(inner / 2));
            t[i] = oskar_grid_function_spheroidal(fabs(nu));
        }
    }

    /* Allocate space for the kernels. */
    kernel_cube = oskar_mem_create(prec | OSKAR_COMPLEX, OSKAR_CPU,
            ((size_t) num_w_planes) * kernel_plane_size, status);

    /* Create scratch arrays and FFT plan for the phase screens. */
    screen = oskar_mem_create(prec | OSKAR_COMPLEX,
            OSKAR_CPU, conv_size * conv_size, status);
    screen_ptr = screen;
    const int fft_loc = (h->generate_w_kernels_on_gpu && h->num_gpus > 0) ?
            h->dev_loc : OSKAR_CPU;
    if (fft_loc != OSKAR_CPU)
    {
        oskar_device_set(h->dev_loc, h->gpu_ids[0], status);
        screen_gpu = oskar_mem_create(prec | OSKAR_COMPLEX,
                h->dev_loc, conv_size * conv_size, status);
        taper_gpu = oskar_mem_create_copy(taper, h->dev_loc, status);
        screen_ptr = screen_gpu;
        taper_ptr = taper_gpu;
    }
    fft = oskar_fft_create(h->imager_prec, fft_loc,
            2, conv_size, 0, status);
    oskar_fft_set_ensure_consistent_norm(fft, 0);

    /* Evaluate kernels. */
    ptr_in = oskar_mem_char(screen);
    const size_t element_size = 2 * oskar_mem_element_size(prec);
    const size_t copy_len = (*conv_size_half) * element_size;
    maxes = (double*) calloc(num_w_planes, sizeof(double));
    for (i = 0; i < num_w_planes; ++i)
    {
        size_t iy = 0, in = 0, out = 0, offset = 0;

        /* Generate the tapered phase screen. */
        oskar_imager_generate_w_phase_screen(i, conv_size, inner,
                sampling, w_scale, taper_ptr, screen_ptr, status);

        /* Perform the FFT to get the kernel. No shifts are required. */
        oskar_fft_exec(fft, screen_ptr, status);
        if (screen_ptr != screen)
        {
            oskar_mem_copy(screen, screen_ptr, status);
        }
        if (*status) break;

        /* Get the maximum (from the first element). */
        if (prec == OSKAR_DOUBLE)
        {
            const double* t = (const double*) oskar_mem_void_const(screen);
            maxes[i] = sqrt(t[0]*t[0] + t[1]*t[1]);
        }
        else
        {
            const float* t = (const float*) oskar_mem_void_const(screen);
            maxes[i] = sqrt(t[0]*t[0] + t[1]*t[1]);
        }

        /* Save only the first quarter of the kernel; the rest is redundant. */
        offset = kernel_plane_size * element_size * (size_t) i;
        ptr_out = oskar_mem_char(kernel_cube) + offset;
        for (iy = 0; iy < *conv_size_half; ++iy)
        {
            memcpy(ptr_out + out, ptr_in + in, copy_len);
            in += element_size * (size_t) conv_size;
            out += copy_len;
        }
    }
    oskar_fft_free(fft);
    oskar_mem_free(screen, status);
    oskar_mem_free(screen_gpu, status);
    oskar_mem_free(taper, status);
    oskar_mem_free(taper_gpu, status);

    /* Get scaling factor needed for normalisation. */
    for (i = 0; i < num_w_planes; ++i) max_val = MAX(max_val, maxes[i]);
    *norm_factor = 1.0 / max_val;
    free(maxes);
    return kernel_cube;
}


static oskar_Mem* oskar_imager_evaluate_w_kernel_support_sizes(
        int num_w_planes, int oversample, size_t conv_size_half,
        const oskar_Mem* kernel_cube, double norm_factor, int* status)
{
    int *supp = 0, i = 0;
    oskar_Mem* w_support = 0;
    if (*status || !kernel_cube) return 0;
    w_support = oskar_mem_create(OSKAR_INT, OSKAR_CPU, num_w_planes, status);
    supp = oskar_mem_int(w_support, status);
    const double threshold = 1e-3 / norm_factor;
    const int prec = oskar_mem_precision(kernel_cube);
    const int conv_size = ((int) conv_size_half + 1) * 2;
    for (i = 0; i < num_w_planes; ++i)
    {
        int found = 0, j = 0;
        if (*status) break;
        const size_t start = conv_size_half * conv_size_half * (size_t) i;
        if (prec == OSKAR_DOUBLE)
        {
            const double *RESTRICT p =
                    (const double*) oskar_mem_void_const(kernel_cube);
            for (j = (int) conv_size_half - 1; j > 0; j--)
            {
                const size_t i1 = ((size_t) j * conv_size_half + start) << 1;
                const size_t i2 = ((size_t) j + start) << 1;
                const double v1 = sqrt(p[i1]*p[i1] + p[i1+1]*p[i1+1]);
                const double v2 = sqrt(p[i2]*p[i2] + p[i2+1]*p[i2+1]);
                if ((v1 > threshold) || (v2 > threshold))
                {
                    found = 1;
                    break;
                }
            }
        }
        else
        {
            const float *RESTRICT p =
                    (const float*) oskar_mem_void_const(kernel_cube);
            for (j = (int) conv_size_half - 1; j > 0; j--)
            {
                const size_t i1 = ((size_t) j * conv_size_half + start) << 1;
                const size_t i2 = ((size_t) j + start) << 1;
                const double v1 = sqrt(p[i1]*p[i1] + p[i1+1]*p[i1+1]);
                const double v2 = sqrt(p[i2]*p[i2] + p[i2+1]*p[i2+1]);
                if ((v1 > threshold) || (v2 > threshold))
                {
                    found = 1;
                    break;
                }
            }
        }
        if (found)
        {
            supp[i] = 1 + (int)(0.5 + (double)j / (double)oversample);
            if (supp[i] * oversample * 2 >= conv_size)
            {
                supp[i] = conv_size / 2 / oversample - 1;
            }
        }
    }
    return w_support;
}


static void oskar_imager_trim_and_save_kernel_cube(oskar_Imager* h,
        int num_w_planes, oskar_Mem* support, size_t* conv_size_half,
        oskar_Mem* kernel_cube, int* status)
{
    char* fname = 0;
    int i = 0, max_val = -INT_MAX;
    const int* supp = 0;
    fitsfile* f = 0;
    char *ttype[] = {"SUPPORT"};
    char *tform[] = {"1J"}; /* 32-bit integer. */
    char *tunit[] = {"\0"};
    char extname[] = "W_KERNELS";
    double cellsize_rad = 0.0, fov_rad = 0.0, w_scale = 0.0;
    int grid_size = 0, image_size = 0, oversample = 0;
    if (*status || !kernel_cube) return;
    supp = oskar_mem_int_const(support, status);
    for (i = 0; i < num_w_planes; ++i) max_val = MAX(max_val, supp[i]);
    const int new_conv_size = 2 * (max_val + 2) * h->oversample;
    const size_t new_conv_size_half = (size_t) new_conv_size / 2 - 2;
    if (new_conv_size_half < *conv_size_half)
    {
        size_t iy = 0, in = 0, out = 0;
        char *ptr = oskar_mem_char(kernel_cube);
        const int prec = oskar_mem_precision(kernel_cube);
        const size_t element_size = 2 * oskar_mem_element_size(prec);
        const size_t copy_len = element_size * new_conv_size_half;
        const size_t kernel_plane_size = (*conv_size_half) * (*conv_size_half);
        for (i = 0; i < num_w_planes; ++i)
        {
            in = kernel_plane_size * element_size * (size_t) i;
            for (iy = 0; iy < new_conv_size_half; ++iy)
            {
                /* Use memmove() rather than memcpy() to allow for overlap. */
                memmove(ptr + out, ptr + in, copy_len);
                in += (*conv_size_half) * element_size;
                out += copy_len;
            }
        }
        *conv_size_half = new_conv_size_half;
        oskar_mem_realloc(kernel_cube,
                ((size_t) num_w_planes) * new_conv_size_half *
                new_conv_size_half, status);
    }

    /* Write kernel cube as primary and secondary HDU. */
    fname = (char*) calloc(80, sizeof(char));
    sprintf(fname, "oskar_w_kernel_cache_%04d.fits", num_w_planes);
    oskar_mem_write_fits_cube(kernel_cube, fname,
            (int) *conv_size_half, (int) *conv_size_half, num_w_planes, -1,
            status);

    /* Write relevant imaging parameters as primary header keywords. */
    fits_open_file(&f, fname, READWRITE, status);
    cellsize_rad = h->cellsize_rad;
    fov_rad = h->fov_deg * M_PI / 180.0;
    w_scale = h->w_scale;
    grid_size = oskar_imager_plane_size(h);
    image_size = h->image_size;
    oversample = h->oversample;
    fits_write_key(f, TINT, "OVERSAMP", &oversample,
            "kernel oversample parameter", status);
    fits_write_key(f, TINT, "GRIDSIZE", &grid_size,
            "grid side length", status);
    fits_write_key(f, TINT, "IMSIZE", &image_size,
            "final image side length, in pixels", status);
    fits_write_key(f, TDOUBLE, "FOV", &fov_rad,
            "final image field of view, in radians", status);
    fits_write_key(f, TDOUBLE, "CELLSIZE", &cellsize_rad,
            "final image cell size, in radians", status);
    fits_write_key(f, TDOUBLE, "W_SCALE", &w_scale,
            "w_scale parameter", status);

    /* Write kernel support sizes as a binary table extension. */
    fits_create_tbl(f, BINARY_TBL, num_w_planes,
            1, ttype, tform, tunit, extname, status);
    fits_write_col(f, TINT, 1, 1, 1, num_w_planes,
            oskar_mem_int(support, status), status);
    fits_close_file(f, status);
    free(fname);
}


static void oskar_imager_normalise_kernel_cube(const oskar_Mem* support,
        int oversample, size_t conv_size_half, oskar_Mem* kernel_cube,
        int* status)
{
    double sum = 0.0;
    int ix = 0, iy = 0;
    if (*status) return;
    const int* supp = oskar_mem_int_const(support, status);

    /* Normalise so that kernel 0 sums to 1,
     * when jumping in steps of oversample. */
    if (oskar_mem_precision(kernel_cube) == OSKAR_DOUBLE)
    {
        const double *RESTRICT p = oskar_mem_double_const(kernel_cube, status);
        for (iy = -supp[0]; iy <= supp[0]; ++iy)
        {
            for (ix = -supp[0]; ix <= supp[0]; ++ix)
            {
                sum += p[2 * (abs(ix) * oversample +
                        conv_size_half * (abs(iy) * oversample))];
            }
        }
    }
    else
    {
        const float *RESTRICT p = oskar_mem_float_const(kernel_cube, status);
        for (iy = -supp[0]; iy <= supp[0]; ++iy)
        {
            for (ix = -supp[0]; ix <= supp[0]; ++ix)
            {
                sum += p[2 * (abs(ix) * oversample +
                        conv_size_half * (abs(iy) * oversample))];
            }
        }
    }
    oskar_mem_scale_real(kernel_cube, 1.0 / sum,
            0, oskar_mem_length(kernel_cube), status);
}


static void oskar_imager_rearrange_kernels(int num_w_planes,
        const oskar_Mem* support, int oversample, size_t conv_size_half,
        const oskar_Mem* kernels_in, oskar_Mem* kernels_out,
        int* rearranged_kernel_start, int* status)
{
    int w = 0, j = 0, k = 0, off_u = 0, off_v = 0;
    size_t rearranged_size = 0;
    float2*  out_f = 0;
    double2* out_d = 0;
    if (*status) return;
    const float2*  in_f = (const float2*)  oskar_mem_void_const(kernels_in);
    const double2* in_d = (const double2*) oskar_mem_void_const(kernels_in);
    const int* supp = oskar_mem_int_const(support, status);
    const int oversample_h = oversample / 2;
    const int prec = oskar_mem_precision(kernels_in);
    const size_t height = oversample_h + 1;

    /* Allocate enough memory for the rearranged kernels. */
    for (w = 0; w < num_w_planes; w++)
    {
        const size_t conv_len = 2 * (size_t)(supp[w]) + 1;
        const size_t width = (oversample_h * conv_len + 1) * conv_len;
        rearranged_kernel_start[w] = (int) rearranged_size;
        rearranged_size += (width * height);
    }
    oskar_mem_realloc(kernels_out, rearranged_size, status);
    /* oskar_mem_set_value_real(kernels_out, 1e9, 0, 0, status); */
    out_f = (float2*)  oskar_mem_void(kernels_out);
    out_d = (double2*) oskar_mem_void(kernels_out);

    for (w = 0; w < num_w_planes; w++)
    {
        const int w_support = supp[w];
        const size_t conv_len = 2 * (size_t)(supp[w]) + 1;
        const size_t width = ((size_t)oversample_h * conv_len + 1) * conv_len;
        const size_t c_in = conv_size_half * conv_size_half * (size_t)w;
        const size_t c_out = (size_t)(rearranged_kernel_start[w]);

        /* Within each kernel, off_u is slowest varying, so if the
         * rearranged kernel is viewed as a very squashed image, the
         * row index to use for off_u is given by width * abs(off_u).
         *
         * "Reverse" rows and rearrange elements in U dimension
         * for reasonable cache utilisation.
         * For idx_v = abs(off_v + j * oversample), the offset from the
         * start of the row is given by width-1 - (idx_v * conv_len).
         * Stride is +1 for positive or zero values of off_u;
         * stride is -1 for negative values of off_u.
         */
        for (off_u = -oversample_h; off_u <= oversample_h; off_u++)
        {
            const size_t mid = c_out + (abs(off_u) + 1) * width - 1 - w_support;
            const int stride = (off_u >= 0) ? 1 : -1;
            for (off_v = -oversample_h; off_v <= oversample_h; off_v++)
            {
                for (j = 0; j <= w_support; j++)
                {
                    const unsigned int idx_v = abs(off_v + j * oversample);
                    const size_t p = mid - (idx_v * conv_len);
                    for (k = -w_support; k <= w_support; k++)
                    {
                        const unsigned int idx_u = abs(off_u + k * oversample);
                        const size_t a = c_in + idx_v * conv_size_half + idx_u;
                        const size_t b = p + stride * k;
                        if (prec == OSKAR_SINGLE)
                        {
                            out_f[b] = in_f[a];
                        }
                        else
                        {
                            out_d[b] = in_d[a];
                        }
                    }
                }
            }
        }
#if 0
        if (w == 0)
        {
            oskar_Mem* a1 = oskar_mem_create_alias(kernels_out, c_out,
                    width * height, status);
            oskar_Mem* a2 = oskar_mem_create_alias(kernels_in, c_in,
                    conv_size_half * conv_size_half, status);
            oskar_mem_write_fits_cube(a1, "kernel_rearranged",
                    (int) width, (int) height, 1, 0, status);
            oskar_mem_write_fits_cube(a2, "kernel_orig",
                    conv_size_half, conv_size_half, 1, 0, status);
            oskar_mem_free(a1, status);
            oskar_mem_free(a2, status);
            printf("At w = %d, w_support = %d\n", w, w_support);
            printf("Original size: %lu. Rearranged size: %lu\n",
                    (unsigned long) oskar_mem_length(kernels_in),
                    (unsigned long) oskar_mem_length(kernels_out));
        }
#endif
    }
#if 0
    FILE* fhan = fopen("rearranged_kernels.txt", "w");
    oskar_mem_save_ascii(fhan, 1, rearranged_size, status, kernels_out);
    fclose(fhan);
#endif
}

#ifdef __cplusplus
}
#endif
