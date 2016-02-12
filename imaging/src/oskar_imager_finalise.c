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

#include <oskar_fftphase.h>
#include <oskar_grid_correction.h>
#include <oskar_imager.h>
#include <oskar_imager_finalise.h>
#include <oskar_imager_reset_cache.h>
#include <oskar_mem.h>
#include <fitsio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static void write_plane(oskar_Imager* h, oskar_Mem* plane,
        int t, int c, int p, int* status);
static void oskar_imager_fft(oskar_Imager* h, oskar_Mem* plane, int* status);


void oskar_imager_finalise(oskar_Imager* h, oskar_Mem* output_plane,
        int* status)
{
    int t, c, p, i = 0;
    if (*status) return;

    /* Finalise all the planes. */
    for (i = 0; i < h->num_planes; ++i)
        oskar_imager_finalise_plane(h, h->planes[i], h->plane_norm[i], status);

    /* Copy plane 0 to output plane if given. */
    if (output_plane && h->num_planes > 0)
    {
        size_t bytes;
        bytes = h->num_pixels * oskar_mem_element_size(h->imager_prec);
        memcpy(oskar_mem_void(output_plane),
                oskar_mem_void_const(h->planes[0]), bytes);
    }

    /* Write to files if required. */
    if (h->fits_file[0])
    {
        for (t = 0, i = 0; t < h->im_num_times; ++t)
        {
            for (c = 0; c < h->im_num_channels; ++c)
            {
                for (p = 0; p < h->im_num_pols; ++p, ++i)
                {
                    write_plane(h, h->planes[i], t, c, p, status);
                    h->plane_norm[i] = 0.0;
                }
            }
        }
    }

    /* Reset imager memory. */
    oskar_imager_reset_cache(h, status);
}


void oskar_imager_finalise_plane(oskar_Imager* h, oskar_Mem* plane,
        double plane_norm, int* status)
{
    if (*status) return;
    oskar_mem_scale_real(plane, 1.0/plane_norm, status);
    if (h->algorithm == OSKAR_ALGORITHM_FFT)
        oskar_imager_fft(h, plane, status);
}


void oskar_imager_fft(oskar_Imager* h, oskar_Mem* plane, int* status)
{
    int i, size, num_pixels;
    DeviceData* d;
    if (*status) return;

    size = h->size;
    num_pixels = size * size;
    cudaSetDevice(h->cuda_device_ids[0]);
    d = &h->d[0];

    /* Make image using FFT. */
    if (oskar_mem_precision(plane) == OSKAR_DOUBLE)
    {
        double *t;
        oskar_fftphase_cd(size, size, oskar_mem_double(plane, status));
        oskar_mem_copy(d->plane_gpu, plane, status);
        cufftExecZ2Z(h->cufft_plan_imager, oskar_mem_void(d->plane_gpu),
                oskar_mem_void(d->plane_gpu), CUFFT_FORWARD);
        oskar_mem_copy(plane, d->plane_gpu, status);
        oskar_fftphase_cd(size, size, oskar_mem_double(plane, status));
        t = oskar_mem_double(plane, status); /* Get the real part. */
        for (i = 0; i < num_pixels; ++i) t[i] = t[2 * i];
        oskar_grid_correction_d(size,
                oskar_mem_double(h->corr_func, status), t);
    }
    else
    {
        float *t;
        oskar_fftphase_cf(size, size, oskar_mem_float(plane, status));
        oskar_mem_copy(d->plane_gpu, plane, status);
        cufftExecC2C(h->cufft_plan_imager, oskar_mem_void(d->plane_gpu),
                oskar_mem_void(d->plane_gpu), CUFFT_FORWARD);
        oskar_mem_copy(plane, d->plane_gpu, status);
        oskar_fftphase_cf(size, size, oskar_mem_float(plane, status));
        t = oskar_mem_float(plane, status); /* Get the real part. */
        for (i = 0; i < num_pixels; ++i) t[i] = t[2 * i];
        oskar_grid_correction_f(size,
                oskar_mem_double(h->corr_func, status), t);
    }
}


void write_plane(oskar_Imager* h, oskar_Mem* plane,
        int t, int c, int p, int* status)
{
    int datatype;
    long firstpix[4];
    if (*status) return;
    datatype = (oskar_mem_is_double(plane) ? TDOUBLE : TFLOAT);
    firstpix[0] = 1;
    firstpix[1] = 1;
    firstpix[2] = 1 + c;
    firstpix[3] = 1 + t;
    fits_write_pix(h->fits_file[p], datatype, firstpix, h->num_pixels,
            oskar_mem_void(plane), status);
}


#ifdef __cplusplus
}
#endif
