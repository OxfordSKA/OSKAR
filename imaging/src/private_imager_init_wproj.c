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

#include <cufft.h>
#include <cuda_runtime_api.h>

#include <private_imager.h>
#include <private_imager_free_wproj.h>
#include <private_imager_init_wproj.h>
#include <oskar_convert_fov_to_cellsize.h>
#include <oskar_fftpack_cfft.h>
#include <oskar_fftpack_cfft_f.h>
#include <oskar_get_memory_usage.h>
#include <oskar_grid_functions_spheroidal.h>

#include <oskar_cmath.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

static int oskar_composite_nearest_even(int value, int* smaller, int *larger);
static void generate_w_phase_screen(const int iw, const int conv_size,
        const int inner, const double sampling, const double w_scale,
        const double* taper_func, oskar_Mem* screen, int* status);

/*
 * W-kernel generation is based on CASA implementation
 * in code/synthesis/TransformMachines/WPConvFunc.cc
 */
void oskar_imager_init_wproj(oskar_Imager* h, int* status)
{
    size_t max_mem_bytes, max_bytes_per_plane;
    int i, iw, ix, iy, *supp, new_conv_size, oversample;
    int conv_size, conv_size_half, inner, nearest;
    double l_max, max_conv_size, max_uvw, max_val, sampling, sum, ww_mid;
    double *taper_func, *maxes;
    double image_padding = 1.2;
    cufftHandle cufft_plan = 0;
    oskar_Mem *screen = 0, *screen_gpu = 0, *wsave = 0, *work = 0;
    int fft_on_gpu = 1;

    /* Clear old kernels. */
    oskar_imager_free_wproj(h, status);
    if (*status) return;

    /* Get GCF padding oversample factor. */
    oversample = h->oversample;

    /* Calculate cellsize. */
    h->cellsize_rad = oskar_convert_fov_to_cellsize(h->fov_deg * M_PI/180,
            h->image_size);

    /* Calculate number of w-planes if not set. */
    if (h->num_w_planes < 1)
    {
        max_uvw = 1.05 * h->ww_max;
        ww_mid = 0.5 * (h->ww_min + h->ww_max);
        if (h->ww_rms > ww_mid)
            max_uvw *= h->ww_rms / ww_mid;
        h->num_w_planes = (int)(max_uvw *
                fabs(sin(h->cellsize_rad * h->image_size / 2.0)));
    }
    else
    {
        if (h->ww_max > 0.0)
            max_uvw = 1.05 * h->ww_max;
        else
            max_uvw = 0.25 / fabs(h->cellsize_rad);
    }

    /* Calculate convolution kernel size. */
    h->w_scale = pow(h->num_w_planes - 1, 2.0) / max_uvw;
    max_mem_bytes = oskar_get_total_physical_memory();
    max_bytes_per_plane = 64 * 1024 * 1024; /* 64 MB per plane */
    max_mem_bytes = MIN(max_mem_bytes, max_bytes_per_plane * h->num_w_planes);
    max_conv_size = sqrt(max_mem_bytes / (16.0 * h->num_w_planes));
    nearest = oskar_composite_nearest_even(2 * (int)(max_conv_size / 2.0), 0, 0);
    conv_size = MIN((int)(h->image_size * image_padding), nearest);
    conv_size_half = conv_size / 2 - 1;
    h->conv_size_half = conv_size_half;

    /* Allocate kernels and support array. */
    h->w_support = oskar_mem_create(OSKAR_INT, OSKAR_CPU,
            h->num_w_planes, status);
    h->w_kernels = oskar_mem_create(h->imager_prec | OSKAR_COMPLEX, OSKAR_CPU,
            ((size_t) h->num_w_planes) * ((size_t) conv_size_half) *
            ((size_t) conv_size_half), status);
    supp = oskar_mem_int(h->w_support, status);
    if (*status) return;

    /* Get size of inner region of kernel and padded grid size. */
    inner = conv_size / oversample;
    l_max = sin(0.5 * h->fov_deg * M_PI/180.0);
    sampling = (2.0 * l_max) / h->image_size;
    sampling *= oversample;
    (void) oskar_composite_nearest_even(
            image_padding * ((double)(h->image_size)) - 0.5, 0, &h->grid_size);
    sampling *= ((double) h->grid_size) / ((double) conv_size);

    /* Create scratch arrays and FFT plan for the phase screens. */
    screen = oskar_mem_create(h->imager_prec | OSKAR_COMPLEX,
            OSKAR_CPU, conv_size * conv_size, status);
    if (fft_on_gpu)
    {
        screen_gpu = oskar_mem_create(h->imager_prec | OSKAR_COMPLEX,
                OSKAR_GPU, conv_size * conv_size, status);
        if (h->imager_prec == OSKAR_DOUBLE)
            cufftPlan2d(&cufft_plan, conv_size, conv_size, CUFFT_Z2Z);
        else
            cufftPlan2d(&cufft_plan, conv_size, conv_size, CUFFT_C2C);
    }
    else
    {
        int len_save = 4 * conv_size +
                2 * (int)(log((double)conv_size) / log(2.0)) + 8;
        wsave = oskar_mem_create(h->imager_prec, OSKAR_CPU, len_save, status);
        work = oskar_mem_create(h->imager_prec, OSKAR_CPU,
                2 * conv_size * conv_size, status);
        if (h->imager_prec == OSKAR_DOUBLE)
            oskar_fftpack_cfft2i(conv_size, conv_size,
                    oskar_mem_double(wsave, status));
        else
            oskar_fftpack_cfft2i_f(conv_size, conv_size,
                    oskar_mem_float(wsave, status));
    }

    /* Generate 1D spheroidal tapering function to cover the inner region. */
    taper_func = (double*) calloc(inner, sizeof(double));
    for (i = 0; i < inner; ++i)
    {
        double nu;
        nu = (i - (inner / 2)) / ((double)(inner / 2));
        taper_func[i] = oskar_grid_function_spheroidal(fabs(nu));
    }

    /* Evaluate kernels. */
    maxes = (double*) calloc(h->num_w_planes, sizeof(double));
    for (iw = 0; iw < h->num_w_planes; ++iw)
    {
        int ind1, ind2, offset;

        /* Generate the tapered phase screen. */
        generate_w_phase_screen(iw, conv_size, inner, sampling,
                h->w_scale, taper_func, screen, status);

        /* Perform the FFT to get the kernel. No shifts are required. */
        if (fft_on_gpu)
        {
            oskar_mem_copy(screen_gpu, screen, status);
            if (oskar_mem_precision(screen) == OSKAR_DOUBLE)
                cufftExecZ2Z(cufft_plan, oskar_mem_void(screen_gpu),
                        oskar_mem_void(screen_gpu), CUFFT_FORWARD);
            else
                cufftExecC2C(cufft_plan, oskar_mem_void(screen_gpu),
                        oskar_mem_void(screen_gpu), CUFFT_FORWARD);
            oskar_mem_copy(screen, screen_gpu, status);
        }
        else
        {
            if (oskar_mem_precision(screen) == OSKAR_DOUBLE)
                oskar_fftpack_cfft2f(conv_size, conv_size, conv_size,
                        oskar_mem_double(screen, status),
                        oskar_mem_double(wsave, status),
                        oskar_mem_double(work, status));
            else
                oskar_fftpack_cfft2f_f(conv_size, conv_size, conv_size,
                        oskar_mem_float(screen, status),
                        oskar_mem_float(wsave, status),
                        oskar_mem_float(work, status));
        }

        /* Save only the first quarter of the kernel; the rest is redundant. */
        offset = iw * conv_size_half * conv_size_half;
        if (oskar_mem_precision(h->w_kernels) == OSKAR_DOUBLE)
        {
            double *ptr, *scr;
            ptr = oskar_mem_double(h->w_kernels, status);
            scr = oskar_mem_double(screen, status);
            maxes[iw] = sqrt(scr[0]*scr[0] + scr[1]*scr[1]);
            for (iy = 0; iy < conv_size_half; ++iy)
            {
                for (ix = 0; ix < conv_size_half; ++ix)
                {
                    ind1 = 2 * (offset + iy * conv_size_half + ix);
                    ind2 = 2 * (conv_size * iy + ix);
                    ptr[ind1 + 0] = scr[ind2 + 0];
                    ptr[ind1 + 1] = scr[ind2 + 1];
                }
            }
        }
        else
        {
            float *ptr, *scr;
            ptr = oskar_mem_float(h->w_kernels, status);
            scr = oskar_mem_float(screen, status);
            maxes[iw] = sqrt(scr[0]*scr[0] + scr[1]*scr[1]);
            for (iy = 0; iy < conv_size_half; ++iy)
            {
                for (ix = 0; ix < conv_size_half; ++ix)
                {
                    ind1 = 2 * (offset + iy * conv_size_half + ix);
                    ind2 = 2 * (conv_size * iy + ix);
                    ptr[ind1 + 0] = (float) (scr[ind2 + 0]);
                    ptr[ind1 + 1] = (float) (scr[ind2 + 1]);
                }
            }
        }
    }

    /* Clean up. */
    free(taper_func);
    cufftDestroy(cufft_plan);
    oskar_mem_free(screen, status);
    oskar_mem_free(screen_gpu, status);
    oskar_mem_free(wsave, status);
    oskar_mem_free(work, status);

    /* Normalise each plane by the maximum. */
    max_val = -INT_MAX;
    for (iw = 0; iw < h->num_w_planes; ++iw) max_val = MAX(max_val, maxes[iw]);
    oskar_mem_scale_real(h->w_kernels, 1.0 / max_val, status);
    free(maxes);

    /* Find the support size of each kernel by stepping in from the edge. */
    for (iw = 0; iw < h->num_w_planes; ++iw)
    {
        int trial = 0, found = 0, plane_offset, ind1, ind2;
        plane_offset = conv_size_half * conv_size_half * iw;
        if (oskar_mem_precision(h->w_kernels) == OSKAR_DOUBLE)
        {
            double *ptr, v1, v2;
            ptr = oskar_mem_double(h->w_kernels, status);
            for (trial = conv_size_half - 1; trial > 0; trial--)
            {
                ind1 = 2 * (trial * conv_size_half + plane_offset);
                ind2 = 2 * (trial + plane_offset);
                v1 = sqrt(ptr[ind1]*ptr[ind1] + ptr[ind1+1]*ptr[ind1+1]);
                v2 = sqrt(ptr[ind2]*ptr[ind2] + ptr[ind2+1]*ptr[ind2+1]);
                if ((v1 > 1e-3) || (v2 > 1e-3))
                {
                    found = 1;
                    break;
                }
            }
        }
        else
        {
            float *ptr, v1, v2;
            ptr = oskar_mem_float(h->w_kernels, status);
            for (trial = conv_size_half - 1; trial > 0; trial--)
            {
                ind1 = 2 * (trial * conv_size_half + plane_offset);
                ind2 = 2 * (trial + plane_offset);
                v1 = sqrt(ptr[ind1]*ptr[ind1] + ptr[ind1+1]*ptr[ind1+1]);
                v2 = sqrt(ptr[ind2]*ptr[ind2] + ptr[ind2+1]*ptr[ind2+1]);
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

    /* Compact the kernels if we can. */
    max_val = -INT_MAX;
    for (iw = 0; iw < h->num_w_planes; ++iw) max_val = MAX(max_val, supp[iw]);
    new_conv_size = 2 * (max_val + 2) * oversample;
    if (new_conv_size < conv_size)
    {
        char *ptr;
        int new_conv_size_half;
        size_t in = 0, out = 0, copy_len = 0, element_size = 0;
        ptr = oskar_mem_char(h->w_kernels);
        element_size = oskar_mem_element_size(oskar_mem_type(h->w_kernels));
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
        double *ptr = oskar_mem_double(h->w_kernels, status);
        for (iy = -supp[0]; iy <= supp[0]; ++iy)
            for (ix = -supp[0]; ix <= supp[0]; ++ix)
                sum += ptr[2 * (abs(ix) * oversample +
                        conv_size_half * (abs(iy) * oversample))];
    }
    else
    {
        float *ptr = oskar_mem_float(h->w_kernels, status);
        for (iy = -supp[0]; iy <= supp[0]; ++iy)
            for (ix = -supp[0]; ix <= supp[0]; ++ix)
                sum += ptr[2 * (abs(ix) * oversample +
                        conv_size_half * (abs(iy) * oversample))];
    }
    oskar_mem_scale_real(h->w_kernels, 1.0 / sum, status);

    /* Generate grid correction function. */
    h->corr_func = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU,
            h->grid_size, status);
    oskar_grid_correction_function_spheroidal(h->grid_size, oversample,
            oskar_mem_double(h->corr_func, status));

    /* Set up the FFT. */
    if (h->fft_on_gpu)
    {
        /* Generate FFT plan. */
        if (h->imager_prec == OSKAR_DOUBLE)
            cufftPlan2d(&h->cufft_plan, h->grid_size, h->grid_size, CUFFT_Z2Z);
        else
            cufftPlan2d(&h->cufft_plan, h->grid_size, h->grid_size, CUFFT_C2C);
    }
    else
    {
        /* Initialise workspaces for CPU FFT algorithm. */
        int len_save = 4 * h->grid_size +
                2 * (int)(log((double)h->grid_size) / log(2.0)) + 8;
        h->fftpack_wsave = oskar_mem_create(h->imager_prec, OSKAR_CPU,
                len_save, status);
        h->fftpack_work = oskar_mem_create(h->imager_prec, OSKAR_CPU,
                2 * h->grid_size * h->grid_size, status);
        if (h->imager_prec == OSKAR_DOUBLE)
            oskar_fftpack_cfft2i(h->grid_size, h->grid_size,
                    oskar_mem_double(h->fftpack_wsave, status));
        else
            oskar_fftpack_cfft2i_f(h->grid_size, h->grid_size,
                    oskar_mem_float(h->fftpack_wsave, status));
    }
}


static void generate_w_phase_screen(const int iw, const int conv_size,
        const int inner, const double sampling, const double w_scale,
        const double* taper_func, oskar_Mem* screen, int* status)
{
    int ix, iy, ind, offset, inner_half;
    double l, m, msq, phase, rsq, f, taper, taper_x, taper_y;

    oskar_mem_clear_contents(screen, status);
    if (*status) return;
    f = (2.0 * M_PI * iw * iw) / w_scale;
    inner_half = inner / 2;
    if (oskar_mem_precision(screen) == OSKAR_SINGLE)
    {
        float* scr = oskar_mem_float(screen, status);
        for (iy = -inner_half; iy < inner_half; ++iy)
        {
            taper_y = taper_func[iy + inner_half];
            m = sampling * (double)iy;
            msq = m*m;
            offset = (iy > -1 ? iy : (iy + conv_size)) * conv_size;
            for (ix = -inner_half; ix < inner_half; ++ix)
            {
                l = sampling * (double)ix;
                rsq = l*l + msq;
                if (rsq < 1.0)
                {
                    taper_x = taper_func[ix + inner_half];
                    taper = taper_x * taper_y;
                    ind = 2 * (offset + (ix > -1 ? ix : (ix + conv_size)));
                    phase = f * (sqrt(1.0 - rsq) - 1.0);
                    scr[ind]     = taper * cos(phase);
                    scr[ind + 1] = taper * sin(phase);
                }
            }
        }
    }
    else
    {
        double* scr = oskar_mem_double(screen, status);
        for (iy = -inner_half; iy < inner_half; ++iy)
        {
            taper_y = taper_func[iy + inner_half];
            m = sampling * (double)iy;
            msq = m*m;
            offset = (iy > -1 ? iy : (iy + conv_size)) * conv_size;
            for (ix = -inner_half; ix < inner_half; ++ix)
            {
                l = sampling * (double)ix;
                rsq = l*l + msq;
                if (rsq < 1.0)
                {
                    taper_x = taper_func[ix + inner_half];
                    taper = taper_x * taper_y;
                    ind = 2 * (offset + (ix > -1 ? ix : (ix + conv_size)));
                    phase = f * (sqrt(1.0 - rsq) - 1.0);
                    scr[ind]     = taper * cos(phase);
                    scr[ind + 1] = taper * sin(phase);
                }
            }
        }
    }
}


static int cmpfunc(const void* a, const void* b)
{
    return (*(const int*)a - *(const int*)b);
}


static int oskar_composite_nearest_even(int value, int* smaller, int *larger)
{
    double x;
    int i = 0, i2, i3, i5, n2, n3, n5, nt, *values, up, down;
    x = (double) value;
    n2 = 1 + (int) (log(x) / log(2.0) + 1.0);
    n3 = 1 + (int) (log(x) / log(3.0) + 1.0);
    n5 = 1 + (int) (log(x) / log(5.0) + 1.0);
    nt = n2 * n3 * n5;
    values = (int*) malloc(nt * sizeof(int));
    for (i2 = 0; i2 < n2; ++i2)
    {
        for (i3 = 0; i3 < n3; ++i3)
        {
            for (i5 = 0; i5 < n5; ++i5, ++i)
            {
                values[i] = (int) round(
                        pow(2.0, (double) i2) *
                        pow(3.0, (double) i3) *
                        pow(5.0, (double) i5));
            }
        }
    }
    qsort(values, nt, sizeof(int), cmpfunc);

    /* Get next larger even. */
    for (i = 0; i < nt; ++i)
    {
        up = values[i];
        if ((up > value) && (up % 2 == 0)) break;
    }

    /* Get next smaller even. */
    for (i = nt - 1; i >= 0; --i)
    {
        down = values[i];
        if ((down < value) && (down % 2 == 0)) break;
    }

    free(values);
    if (smaller) *smaller = down;
    if (larger) *larger = up;
    return (abs(up - value) < abs(down - value) ? up : down);
}


#ifdef __cplusplus
}
#endif
