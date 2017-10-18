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

#ifdef OSKAR_HAVE_CUDA
#include <cufft.h>
#endif

#include "imager/private_imager.h"
#include "imager/oskar_imager.h"

#include "imager/private_imager_composite_nearest_even.h"
#include "imager/private_imager_generate_w_phase_screen.h"
#include "imager/private_imager_init_wproj.h"
#include "imager/oskar_grid_functions_spheroidal.h"
#include "math/oskar_cmath.h"
#include "math/oskar_fftpack_cfft.h"
#include "math/oskar_fftpack_cfft_f.h"
#include "utility/oskar_get_memory_usage.h"
#include "utility/oskar_device_utils.h"

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define SAVE_KERNELS 0

/*
 * W-kernel generation is based on CASA implementation
 * in code/synthesis/TransformMachines/WPConvFunc.cc
 */
void oskar_imager_init_wproj(oskar_Imager* h, int* status)
{
    size_t max_mem_bytes, max_bytes_per_plane, element_size, copy_len;
    int i, iw, ix, iy, *supp, new_conv_size, oversample, prec;
    int conv_size, conv_size_half, inner, nearest;
    double l_max, max_conv_size, max_uvw, max_val, sampling, sum;
    double *maxes;
#ifdef OSKAR_HAVE_CUDA
    cufftHandle cufft_plan = 0;
#endif
    oskar_Mem *screen = 0, *screen_gpu = 0, *screen_ptr = 0;
    oskar_Mem *taper = 0, *taper_gpu = 0, *taper_ptr = 0;
    oskar_Mem *wsave = 0, *work = 0;
    char *ptr_out, *ptr_in, *fname = 0;
    if (*status) return;

    /* Get GCF padding oversample factor and imager precision. */
    oversample = h->oversample;
    prec = h->imager_prec;

    /* Calculate required number of w-planes if not set. */
    if (h->ww_max > 0.0)
    {
        double ww_mid;
        max_uvw = 1.05 * h->ww_max;
        ww_mid = 0.5 * (h->ww_min + h->ww_max);
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
    max_mem_bytes = oskar_get_total_physical_memory();
    max_bytes_per_plane = 64 * 1024 * 1024; /* 64 MB per plane */
    max_mem_bytes = MIN(max_mem_bytes, max_bytes_per_plane * h->num_w_planes);
    max_conv_size = sqrt(max_mem_bytes / (16.0 * h->num_w_planes));
    nearest = oskar_imager_composite_nearest_even(
            2 * (int)(max_conv_size / 2.0), 0, 0);
    conv_size = MIN((int)(h->image_size * h->image_padding), nearest);
    conv_size_half = conv_size / 2 - 1;
    h->conv_size_half = conv_size_half;

    /* Allocate kernels and support array. */
    oskar_mem_free(h->w_kernels, status);
    oskar_mem_free(h->w_support, status);
    h->w_support = oskar_mem_create(OSKAR_INT, OSKAR_CPU,
            h->num_w_planes, status);
    h->w_kernels = oskar_mem_create(prec | OSKAR_COMPLEX, OSKAR_CPU,
            ((size_t) h->num_w_planes) * ((size_t) conv_size_half) *
            ((size_t) conv_size_half), status);
    supp = oskar_mem_int(h->w_support, status);
    element_size = oskar_mem_element_size(oskar_mem_type(h->w_kernels));
    if (*status) return;

    /* Get size of inner region of kernel and padded grid size. */
    inner = conv_size / oversample;
    l_max = sin(0.5 * h->fov_deg * M_PI/180.0);
    sampling = (2.0 * l_max * oversample) / h->image_size;
    sampling *= ((double) oskar_imager_plane_size(h)) / ((double) conv_size);

    /* Create scratch arrays and FFT plan for the phase screens. */
    screen = oskar_mem_create(prec | OSKAR_COMPLEX,
            OSKAR_CPU, conv_size * conv_size, status);
    screen_ptr = screen;
#ifdef OSKAR_HAVE_CUDA
    if (h->generate_w_kernels_on_gpu && h->num_gpus > 0)
    {
        oskar_device_set(h->gpu_ids[0], status);
        screen_gpu = oskar_mem_create(prec | OSKAR_COMPLEX,
                OSKAR_GPU, conv_size * conv_size, status);
        screen_ptr = screen_gpu;
        if (prec == OSKAR_DOUBLE)
            cufftPlan2d(&cufft_plan, conv_size, conv_size, CUFFT_Z2Z);
        else
            cufftPlan2d(&cufft_plan, conv_size, conv_size, CUFFT_C2C);
    }
    else
#endif
    {
        int len_save = 4 * conv_size +
                2 * (int)(log((double)conv_size) / log(2.0)) + 8;
        wsave = oskar_mem_create(prec, OSKAR_CPU, len_save, status);
        work = oskar_mem_create(prec, OSKAR_CPU,
                2 * conv_size * conv_size, status);
        if (prec == OSKAR_DOUBLE)
            oskar_fftpack_cfft2i(conv_size, conv_size,
                    oskar_mem_double(wsave, status));
        else
            oskar_fftpack_cfft2i_f(conv_size, conv_size,
                    oskar_mem_float(wsave, status));
    }

    /* Generate 1D spheroidal tapering function to cover the inner region. */
    taper = oskar_mem_create(prec, OSKAR_CPU, inner, status);
    taper_ptr = taper;
    if (prec == OSKAR_DOUBLE)
    {
        double* t = oskar_mem_double(taper, status);
        for (i = 0; i < inner; ++i)
        {
            double nu;
            nu = (i - (inner / 2)) / ((double)(inner / 2));
            t[i] = oskar_grid_function_spheroidal(fabs(nu));
        }
    }
    else
    {
        float* t = oskar_mem_float(taper, status);
        for (i = 0; i < inner; ++i)
        {
            double nu;
            nu = (i - (inner / 2)) / ((double)(inner / 2));
            t[i] = oskar_grid_function_spheroidal(fabs(nu));
        }
    }
#ifdef OSKAR_HAVE_CUDA
    if (h->generate_w_kernels_on_gpu && h->num_gpus > 0)
    {
        taper_gpu = oskar_mem_create_copy(taper, OSKAR_GPU, status);
        taper_ptr = taper_gpu;
    }
#endif

    /* Evaluate kernels. */
    ptr_in   = oskar_mem_char(screen);
    copy_len = element_size * conv_size_half;
    maxes = (double*) calloc(h->num_w_planes, sizeof(double));
    for (iw = 0; iw < h->num_w_planes; ++iw)
    {
        size_t in = 0, out = 0, offset;

        /* Generate the tapered phase screen. */
        oskar_imager_generate_w_phase_screen(iw, conv_size, inner, sampling,
                h->w_scale, taper_ptr, screen_ptr, status);
        if (*status) break;

        /* Perform the FFT to get the kernel. No shifts are required. */
#ifdef OSKAR_HAVE_CUDA
        if (h->generate_w_kernels_on_gpu && h->num_gpus > 0)
        {
            if (oskar_mem_precision(screen) == OSKAR_DOUBLE)
                cufftExecZ2Z(cufft_plan, oskar_mem_void(screen_ptr),
                        oskar_mem_void(screen_ptr), CUFFT_FORWARD);
            else
                cufftExecC2C(cufft_plan, oskar_mem_void(screen_ptr),
                        oskar_mem_void(screen_ptr), CUFFT_FORWARD);
            oskar_mem_copy(screen, screen_ptr, status);
        }
        else
#endif
        {
            if (oskar_mem_precision(screen_ptr) == OSKAR_DOUBLE)
                oskar_fftpack_cfft2f(conv_size, conv_size, conv_size,
                        oskar_mem_double(screen_ptr, status),
                        oskar_mem_double(wsave, status),
                        oskar_mem_double(work, status));
            else
                oskar_fftpack_cfft2f_f(conv_size, conv_size, conv_size,
                        oskar_mem_float(screen_ptr, status),
                        oskar_mem_float(wsave, status),
                        oskar_mem_float(work, status));
        }
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
#ifdef OSKAR_HAVE_CUDA
    cufftDestroy(cufft_plan);
#endif
    oskar_mem_free(screen, status);
    oskar_mem_free(screen_gpu, status);
    oskar_mem_free(taper, status);
    oskar_mem_free(taper_gpu, status);
    oskar_mem_free(wsave, status);
    oskar_mem_free(work, status);

    /* Normalise each plane by the maximum. */
    if (*status) return;
    max_val = -INT_MAX;
    for (iw = 0; iw < h->num_w_planes; ++iw) max_val = MAX(max_val, maxes[iw]);
    oskar_mem_scale_real(h->w_kernels, 1.0 / max_val, status);
    free(maxes);

    /* Find the support size of each kernel by stepping in from the edge. */
    for (iw = 0; iw < h->num_w_planes; ++iw)
    {
        int trial = 0, found = 0, plane_offset, ind1, ind2;
        double v1, v2;
        if (*status) break;
        plane_offset = conv_size_half * conv_size_half * iw;
        if (oskar_mem_precision(h->w_kernels) == OSKAR_DOUBLE)
        {
            const double *restrict p = oskar_mem_double_const(
                    h->w_kernels, status);
            for (trial = conv_size_half - 1; trial > 0; trial--)
            {
                ind1 = 2 * (trial * conv_size_half + plane_offset);
                ind2 = 2 * (trial + plane_offset);
                v1 = sqrt(p[ind1]*p[ind1] + p[ind1+1]*p[ind1+1]);
                v2 = sqrt(p[ind2]*p[ind2] + p[ind2+1]*p[ind2+1]);
                if ((v1 > 1e-3) || (v2 > 1e-3))
                {
                    found = 1;
                    break;
                }
            }
        }
        else
        {
            const float *restrict p = oskar_mem_float_const(
                    h->w_kernels, status);
            for (trial = conv_size_half - 1; trial > 0; trial--)
            {
                ind1 = 2 * (trial * conv_size_half + plane_offset);
                ind2 = 2 * (trial + plane_offset);
                v1 = sqrt(p[ind1]*p[ind1] + p[ind1+1]*p[ind1+1]);
                v2 = sqrt(p[ind2]*p[ind2] + p[ind2+1]*p[ind2+1]);
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
    new_conv_size = 2 * (max_val + 2) * oversample;
    if (new_conv_size < conv_size)
    {
        char *ptr;
        int new_conv_size_half;
        size_t in = 0, out = 0;
        ptr = oskar_mem_char(h->w_kernels);
        new_conv_size_half = new_conv_size / 2 - 2;
        copy_len = element_size * new_conv_size_half;

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
        conv_size = new_conv_size;
        h->conv_size_half = conv_size_half = new_conv_size_half;
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
        const double *restrict p = oskar_mem_double_const(h->w_kernels, status);
        for (iy = -supp[0]; iy <= supp[0]; ++iy)
            for (ix = -supp[0]; ix <= supp[0]; ++ix)
                sum += p[2 * (abs(ix) * oversample +
                        conv_size_half * (abs(iy) * oversample))];
    }
    else
    {
        const float *restrict p = oskar_mem_float_const(h->w_kernels, status);
        for (iy = -supp[0]; iy <= supp[0]; ++iy)
            for (ix = -supp[0]; ix <= supp[0]; ++ix)
                sum += p[2 * (abs(ix) * oversample +
                        conv_size_half * (abs(iy) * oversample))];
    }
    oskar_mem_scale_real(h->w_kernels, 1.0 / sum, status);

#if SAVE_KERNELS
    fname = (char*) calloc(20 + (h->input_root ? strlen(h->input_root) : 0), 1);
    sprintf(fname, "%s_KERNELS", h->input_root ? h->input_root : "");
    oskar_mem_write_fits_cube(h->w_kernels, fname,
            conv_size_half, conv_size_half, h->num_w_planes, -1, status);
#endif
    free(fname);
}

#ifdef __cplusplus
}
#endif
