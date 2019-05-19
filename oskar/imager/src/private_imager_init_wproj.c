/*
 * Copyright (c) 2016-2019, The University of Oxford
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
#include "imager/oskar_imager.h"

#include "imager/private_imager_composite_nearest_even.h"
#include "imager/private_imager_generate_w_phase_screen.h"
#include "imager/private_imager_init_wproj.h"
#include "imager/oskar_grid_functions_spheroidal.h"
#include "math/oskar_cmath.h"
#include "math/oskar_fft.h"
#include "utility/oskar_get_memory_usage.h"
#include "utility/oskar_device.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define SAVE_KERNELS 0

#if SAVE_KERNELS
#include <fitsio.h>

static void write_kernel_metadata(oskar_Imager* h,
        const char* fname, int* status)
{
    fitsfile* f = 0;
    char *ttype[] = {"SUPPORT"};
    char *tform[] = {"1J"}; /* 32-bit integer. */
    char *tunit[] = {"\0"};
    char extname[] = "W_KERNELS";
    double fov_rad;
    int grid_size;
    fits_open_file(&f, fname, READWRITE, status);

    /* Write relevant imaging parameters as primary header keywords. */
    fov_rad = h->fov_deg * M_PI / 180.0;
    grid_size = oskar_imager_plane_size(h);
    fits_write_key(f, TINT, "OVERSAMP", &h->oversample,
            "kernel oversample parameter", status);
    fits_write_key(f, TINT, "GRIDSIZE", &grid_size,
            "grid side length", status);
    fits_write_key(f, TINT, "IMSIZE", &h->image_size,
            "final image side length, in pixels", status);
    fits_write_key(f, TDOUBLE, "FOV", &fov_rad,
            "final image field of view, in radians", status);
    fits_write_key(f, TDOUBLE, "CELLSIZE", &h->cellsize_rad,
            "final image cell size, in radians", status);
    fits_write_key(f, TDOUBLE, "W_SCALE", &h->w_scale,
            "w_scale parameter", status);

    /* Write kernel support sizes as a binary table extension. */
    fits_create_tbl(f, BINARY_TBL, h->num_w_planes,
            1, ttype, tform, tunit, extname, status);
    fits_write_col(f, TINT, 1, 1, 1, h->num_w_planes,
            oskar_mem_int(h->w_support, status), status);
    fits_close_file(f, status);
}
#endif

static void rearrange_kernels(const int num_w_planes, const int* support,
        const int oversample, const int conv_size_half,
        const oskar_Mem* kernels_in, oskar_Mem* kernels_out,
        int* rearranged_kernel_start, int* status);

/*
 * W-kernel generation is based on CASA implementation
 * in code/synthesis/TransformMachines/WPConvFunc.cc
 */
void oskar_imager_init_wproj(oskar_Imager* h, int* status)
{
    size_t max_mem_bytes;
    int i, iw, ix, iy, *supp;
    double *maxes, max_uvw, max_val, sampling, sum;
    oskar_FFT* fft = 0;
    oskar_Mem *screen = 0, *screen_gpu = 0, *screen_ptr = 0;
    oskar_Mem *taper = 0, *taper_gpu = 0, *taper_ptr = 0;
    char *ptr_out, *ptr_in, *fname = 0;
    if (*status) return;

    /* Get GCF padding oversample factor and imager precision. */
    const int oversample = h->oversample;
    const int prec = h->imager_prec;

    /* Calculate required number of w-planes if not set. */
    if (h->ww_max > 0.0)
    {
        const double ww_mid = 0.5 * (h->ww_min + h->ww_max);
        max_uvw = 1.05 * h->ww_max;
        if (h->ww_rms > ww_mid)
            max_uvw *= h->ww_rms / ww_mid;
    }
    else
    {
        max_uvw = 0.25 / fabs(h->cellsize_rad);
    }
    if (h->num_w_planes < 1)
        h->num_w_planes = (int)(max_uvw *
                fabs(sin(h->cellsize_rad * h->image_size / 2.0)));
    if (h->num_w_planes < 16)
        h->num_w_planes = 16;

    /* Calculate convolution kernel size. */
    h->w_scale = pow(h->num_w_planes - 1, 2.0) / max_uvw;
    const size_t max_bytes_per_plane = 64 * 1024 * 1024; /* 64 MB/plane */
    max_mem_bytes = oskar_get_total_physical_memory();
    max_mem_bytes = MIN(max_mem_bytes, max_bytes_per_plane * h->num_w_planes);
    const double max_conv_size = sqrt(max_mem_bytes / (16. * h->num_w_planes));
    const int nearest = oskar_imager_composite_nearest_even(
            2 * (int)(max_conv_size / 2.0), 0, 0);
    const int conv_size = MIN((int)(h->image_size * h->image_padding),nearest);
    const int conv_size_half = conv_size / 2 - 1;
    h->conv_size_half = conv_size_half;

    /* Allocate kernels and support array. */
    oskar_mem_free(h->w_kernels, status);
    oskar_mem_free(h->w_support, status);
    oskar_mem_free(h->w_kernels_compact, status);
    oskar_mem_free(h->w_kernel_start, status);
    h->w_support = oskar_mem_create(OSKAR_INT, OSKAR_CPU,
            h->num_w_planes, status);
    h->w_kernel_start = oskar_mem_create(OSKAR_INT, OSKAR_CPU,
            h->num_w_planes, status);
    h->w_kernels = oskar_mem_create(prec | OSKAR_COMPLEX, OSKAR_CPU,
            ((size_t) h->num_w_planes) * ((size_t) conv_size_half) *
            ((size_t) conv_size_half), status);
    h->w_kernels_compact = oskar_mem_create(prec | OSKAR_COMPLEX, OSKAR_CPU,
            0, status);
    supp = oskar_mem_int(h->w_support, status);
    const size_t element_size = oskar_mem_element_size(prec | OSKAR_COMPLEX);
    if (*status) return;

    /* Get size of inner region of kernel and padded grid size. */
    const int inner = conv_size / oversample;
    const double l_max = sin(0.5 * h->fov_deg * M_PI/180.0);
    sampling = (2.0 * l_max * oversample) / h->image_size;
    sampling *= ((double) oskar_imager_plane_size(h)) / ((double) conv_size);

    /* Create scratch arrays and FFT plan for the phase screens. */
    screen = oskar_mem_create(prec | OSKAR_COMPLEX,
            OSKAR_CPU, conv_size * conv_size, status);
    screen_ptr = screen;
    const int fft_loc = (h->generate_w_kernels_on_gpu && h->num_gpus > 0) ?
            OSKAR_GPU : OSKAR_CPU;
    if (fft_loc != OSKAR_CPU)
    {
        oskar_device_set(h->dev_loc, h->gpu_ids[0], status);
        screen_gpu = oskar_mem_create(prec | OSKAR_COMPLEX,
                h->dev_loc, conv_size * conv_size, status);
        screen_ptr = screen_gpu;
    }
    fft = oskar_fft_create(h->imager_prec, fft_loc, 2, conv_size, 0, status);
    oskar_fft_set_ensure_consistent_norm(fft, 0);

    /* Generate 1D spheroidal tapering function to cover the inner region. */
    taper = oskar_mem_create(prec, OSKAR_CPU, inner, status);
    taper_ptr = taper;
    if (prec == OSKAR_DOUBLE)
    {
        double* t = oskar_mem_double(taper, status);
        for (i = 0; i < inner; ++i)
        {
            const double nu = (i - (inner / 2)) / ((double)(inner / 2));
            t[i] = oskar_grid_function_spheroidal(fabs(nu));
        }
    }
    else
    {
        float* t = oskar_mem_float(taper, status);
        for (i = 0; i < inner; ++i)
        {
            const double nu = (i - (inner / 2)) / ((double)(inner / 2));
            t[i] = oskar_grid_function_spheroidal(fabs(nu));
        }
    }
#ifdef OSKAR_HAVE_CUDA
    if (h->generate_w_kernels_on_gpu && h->num_gpus > 0)
    {
        taper_gpu = oskar_mem_create_copy(taper, h->dev_loc, status);
        taper_ptr = taper_gpu;
    }
#endif

    /* Evaluate kernels. */
    ptr_in = oskar_mem_char(screen);
    const size_t copy_len = element_size * conv_size_half;
    maxes = (double*) calloc(h->num_w_planes, sizeof(double));
    for (iw = 0; iw < h->num_w_planes; ++iw)
    {
        size_t in = 0, out = 0, offset;

        /* Generate the tapered phase screen. */
        oskar_imager_generate_w_phase_screen(iw, conv_size, inner, sampling,
                h->w_scale, taper_ptr, screen_ptr, status);
        if (*status) break;

        /* Perform the FFT to get the kernel. No shifts are required. */
        oskar_fft_exec(fft, screen_ptr, status);
        if (screen_ptr != screen)
            oskar_mem_copy(screen, screen_ptr, status);
        if (*status) break;

        /* Get the maximum (from the first element). */
        if (oskar_mem_precision(screen) == OSKAR_DOUBLE)
        {
            const double* t = (const double*) oskar_mem_void_const(screen);
            maxes[iw] = sqrt(t[0]*t[0] + t[1]*t[1]);
        }
        else
        {
            const float* t = (const float*) oskar_mem_void_const(screen);
            maxes[iw] = sqrt(t[0]*t[0] + t[1]*t[1]);
        }

        /* Save only the first quarter of the kernel; the rest is redundant. */
        offset = iw * conv_size_half * conv_size_half * element_size;
        ptr_out = oskar_mem_char(h->w_kernels) + offset;
        for (iy = 0; iy < conv_size_half; ++iy)
        {
            memcpy(ptr_out + out, ptr_in + in, copy_len);
            in += conv_size * element_size;
            out += copy_len;
        }
    }

    /* Clean up. */
    oskar_fft_free(fft);
    oskar_mem_free(screen, status);
    oskar_mem_free(screen_gpu, status);
    oskar_mem_free(taper, status);
    oskar_mem_free(taper_gpu, status);

    /* Normalise each plane by the maximum. */
    if (*status) return;
    max_val = -INT_MAX;
    for (iw = 0; iw < h->num_w_planes; ++iw) max_val = MAX(max_val, maxes[iw]);
    oskar_mem_scale_real(h->w_kernels, 1.0 / max_val,
            0, oskar_mem_length(h->w_kernels), status);
    free(maxes);

    /* Find the support size of each kernel by stepping in from the edge. */
    for (iw = 0; iw < h->num_w_planes; ++iw)
    {
        int trial = 0, found = 0;
        if (*status) break;
        const int plane_offset = conv_size_half * conv_size_half * iw;
        if (oskar_mem_precision(h->w_kernels) == OSKAR_DOUBLE)
        {
            const double *RESTRICT p = oskar_mem_double_const(
                    h->w_kernels, status);
            for (trial = conv_size_half - 1; trial > 0; trial--)
            {
                const int i1 = 2 * (trial * conv_size_half + plane_offset);
                const int i2 = 2 * (trial + plane_offset);
                const double v1 = sqrt(p[i1]*p[i1] + p[i1+1]*p[i1+1]);
                const double v2 = sqrt(p[i2]*p[i2] + p[i2+1]*p[i2+1]);
                if ((v1 > 1e-3) || (v2 > 1e-3))
                {
                    found = 1;
                    break;
                }
            }
        }
        else
        {
            const float *RESTRICT p = oskar_mem_float_const(
                    h->w_kernels, status);
            for (trial = conv_size_half - 1; trial > 0; trial--)
            {
                const int i1 = 2 * (trial * conv_size_half + plane_offset);
                const int i2 = 2 * (trial + plane_offset);
                const double v1 = sqrt(p[i1]*p[i1] + p[i1+1]*p[i1+1]);
                const double v2 = sqrt(p[i2]*p[i2] + p[i2+1]*p[i2+1]);
                if ((v1 > 1e-3) || (v2 > 1e-3))
                {
                    found = 1;
                    break;
                }
            }
        }
        if (found)
        {
            supp[iw] = 1 + (int)(0.5 + (double)trial / (double)oversample);
            if (supp[iw] * oversample * 2 >= conv_size)
                supp[iw] = conv_size / 2 / oversample - 1;
        }
    }
    if (*status) return;

    /* Compact the kernels if we can. */
    max_val = -INT_MAX;
    for (iw = 0; iw < h->num_w_planes; ++iw) max_val = MAX(max_val, supp[iw]);
    const int new_conv_size = 2 * (max_val + 2) * oversample;
    if (new_conv_size < conv_size)
    {
        size_t in = 0, out = 0;
        char *ptr = oskar_mem_char(h->w_kernels);
        const int new_conv_size_half = new_conv_size / 2 - 2;
        const size_t copy_len = element_size * new_conv_size_half;
        for (iw = 0; iw < h->num_w_planes; ++iw)
        {
            in = iw * conv_size_half * conv_size_half * element_size;
            for (iy = 0; iy < new_conv_size_half; ++iy)
            {
                /* Use memmove() rather than memcpy() to allow for overlap. */
                memmove(ptr + out, ptr + in, copy_len);
                in += conv_size_half * element_size;
                out += copy_len;
            }
        }
        h->conv_size_half = new_conv_size_half;
        oskar_mem_realloc(h->w_kernels,
                ((size_t) h->num_w_planes) * ((size_t) new_conv_size_half) *
                ((size_t) new_conv_size_half), status);
    }
    if (*status) return;

#if 0
    /* Print kernel support sizes. */
    for (iw = 0; iw < h->num_w_planes; ++iw)
    {
        int *supp = oskar_mem_int(h->w_support, status);
        printf("Plane %d, support: %d\n", iw, supp[iw] * oversample);
    }
#endif

    /* Normalise so that kernel 0 sums to 1,
     * when jumping in steps of oversample. */
    sum = 0.0; /* Real part only. */
    if (oskar_mem_precision(h->w_kernels) == OSKAR_DOUBLE)
    {
        const double *RESTRICT p = oskar_mem_double_const(h->w_kernels, status);
        for (iy = -supp[0]; iy <= supp[0]; ++iy)
            for (ix = -supp[0]; ix <= supp[0]; ++ix)
                sum += p[2 * (abs(ix) * oversample +
                        h->conv_size_half * (abs(iy) * oversample))];
    }
    else
    {
        const float *RESTRICT p = oskar_mem_float_const(h->w_kernels, status);
        for (iy = -supp[0]; iy <= supp[0]; ++iy)
            for (ix = -supp[0]; ix <= supp[0]; ++ix)
                sum += p[2 * (abs(ix) * oversample +
                        h->conv_size_half * (abs(iy) * oversample))];
    }
    oskar_mem_scale_real(h->w_kernels, 1.0 / sum,
            0, oskar_mem_length(h->w_kernels), status);

#if SAVE_KERNELS
    /* Save kernels to a FITS file if necessary. */
    fname = (char*) calloc(20 + (h->input_root ? strlen(h->input_root) : 0), 1);
    sprintf(fname, "%s_KERNELS", h->input_root ? h->input_root : "");
    oskar_mem_write_fits_cube(h->w_kernels, fname,
            h->conv_size_half, h->conv_size_half, h->num_w_planes, -1, status);

    /* Write kernel metadata. */
    sprintf(fname, "%s_KERNELS_REAL.fits", h->input_root ? h->input_root : "");
    write_kernel_metadata(h, fname, status);
    sprintf(fname, "%s_KERNELS_IMAG.fits", h->input_root ? h->input_root : "");
    write_kernel_metadata(h, fname, status);
#endif
    free(fname);

    /* Rearrange the kernels. */
    rearrange_kernels(h->num_w_planes, supp, oversample, h->conv_size_half,
            h->w_kernels, h->w_kernels_compact,
            oskar_mem_int(h->w_kernel_start, status), status);

    /* Initialise device memory if required. */
    if (h->num_gpus > 0)
    {
        if (h->num_devices < h->num_gpus)
            oskar_imager_set_num_devices(h, h->num_gpus);
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
    }
}

static void rearrange_kernels(const int num_w_planes, const int* support,
        const int oversample, const int conv_size_half,
        const oskar_Mem* kernels_in, oskar_Mem* kernels_out,
        int* rearranged_kernel_start, int* status)
{
    int w, j, k, off_u, off_v, rearranged_size = 0;
    float2*  out_f;
    double2* out_d;
    const float2*  in_f = (const float2*)  oskar_mem_void_const(kernels_in);
    const double2* in_d = (const double2*) oskar_mem_void_const(kernels_in);
    const int oversample_h = oversample / 2;
    const int prec = oskar_mem_precision(kernels_in);
    const int height = oversample_h + 1;

    /* Allocate enough memory for the rearranged kernels. */
    for (w = 0; w < num_w_planes; w++)
    {
        const int conv_len = 2 * support[w] + 1;
        const int width = (oversample_h * conv_len + 1) * conv_len;
        rearranged_kernel_start[w] = rearranged_size;
        rearranged_size += (width * height);
    }
    oskar_mem_realloc(kernels_out, (size_t) rearranged_size, status);
    /* oskar_mem_set_value_real(kernels_out, 1e9, 0, 0, status); */
    out_f = (float2*)  oskar_mem_void(kernels_out);
    out_d = (double2*) oskar_mem_void(kernels_out);

    for (w = 0; w < num_w_planes; w++)
    {
        const int w_support = support[w];
        const int conv_len = 2 * support[w] + 1;
        const int width = (oversample_h * conv_len + 1) * conv_len;
        const int c_in = w * conv_size_half * conv_size_half;
        const int c_out = rearranged_kernel_start[w];

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
            const int mid = c_out + (abs(off_u) + 1) * width - 1 - w_support;
            const int stride = (off_u >= 0) ? 1 : -1;
            for (off_v = -oversample_h; off_v <= oversample_h; off_v++)
            {
                for (j = 0; j <= w_support; j++)
                {
                    const int idx_v = abs(off_v + j * oversample);
                    const int p = mid - idx_v * conv_len;
                    for (k = -w_support; k <= w_support; k++)
                    {
                        const int idx_u = abs(off_u + k * oversample);
                        const int a = c_in + idx_v * conv_size_half + idx_u;
                        const int b = p + stride * k;
                        if (prec == OSKAR_SINGLE)
                            out_f[b] = in_f[a];
                        else
                            out_d[b] = in_d[a];
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
