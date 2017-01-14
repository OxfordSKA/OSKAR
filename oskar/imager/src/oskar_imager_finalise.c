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

#ifdef OSKAR_HAVE_CUDA
#include <cufft.h>
#endif

#include "imager/private_imager.h"
#include "imager/oskar_imager.h"

#include "utility/oskar_device_utils.h"
#include "imager/oskar_fftpack_cfft.h"
#include "imager/oskar_fftpack_cfft_f.h"
#include "imager/oskar_fftphase.h"
#include "imager/oskar_grid_correction.h"
#include "mem/oskar_mem.h"
#include <fitsio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static void write_plane(oskar_Imager* h, oskar_Mem* plane,
        int t, int c, int p, int* status);


void oskar_imager_finalise(oskar_Imager* h,
        int num_output_images, oskar_Mem** output_images,
        int num_output_grids, oskar_Mem** output_grids, int* status)
{
    int t, c, p, i;
    if (*status || !h->planes) return;

    /* Adjust normalisation if required. */
    if (h->scale_norm_with_num_input_files)
    {
        for (i = 0; i < h->num_planes; ++i)
            h->plane_norm[i] /= h->num_files;
    }

    /* Copy grids to output grid planes if given. */
    for (i = 0; (i < h->num_planes) && (i < num_output_grids); ++i)
    {
        oskar_mem_copy(output_grids[i], h->planes[i], status);
        oskar_mem_scale_real(output_grids[i], 1.0 / h->plane_norm[i], status);
    }

    /* Check if images are required. */
    if (h->fits_file[0] || output_images)
    {
        /* Finalise all the planes. */
        for (i = 0; i < h->num_planes; ++i)
        {
            oskar_imager_finalise_plane(h,
                    h->planes[i], h->plane_norm[i], status);
            oskar_imager_trim_image(h->planes[i],
                    h->grid_size, h->image_size, status);
        }

        /* Copy images to output image planes if given. */
        for (i = 0; (i < h->num_planes) && (i < num_output_images); ++i)
        {
            memcpy(oskar_mem_void(output_images[i]),
                    oskar_mem_void_const(h->planes[i]), h->image_size *
                    h->image_size * oskar_mem_element_size(h->imager_prec));
        }

        /* Write to files if required. */
        for (t = 0, i = 0; t < h->im_num_times; ++t)
            for (c = 0; c < h->im_num_channels; ++c)
                for (p = 0; p < h->im_num_pols; ++p, ++i)
                    write_plane(h, h->planes[i], t, c, p, status);
    }

    /* Reset imager memory. */
    oskar_imager_reset_cache(h, status);
}


void oskar_imager_finalise_plane(oskar_Imager* h, oskar_Mem* plane,
        double plane_norm, int* status)
{
    int size, num_cells;
    DeviceData* d;
    if (*status) return;

    /* Apply normalisation. */
    if (plane_norm > 0.0 || plane_norm < 0.0)
        oskar_mem_scale_real(plane, 1.0 / plane_norm, status);
    if (h->algorithm == OSKAR_ALGORITHM_DFT_2D ||
            h->algorithm == OSKAR_ALGORITHM_DFT_3D)
        return;

    /* Check plane is complex type, as plane must be gridded visibilities. */
    if (!oskar_mem_is_complex(plane))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Make image using FFT and apply grid correction. */
    size = h->grid_size;
    num_cells = size * size;
    d = &h->d[0];
    if (oskar_mem_precision(plane) == OSKAR_DOUBLE)
    {
        oskar_fftphase_cd(size, size, oskar_mem_double(plane, status));
        if (h->fft_on_gpu)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_device_set(h->cuda_device_ids[0], status);
            oskar_mem_copy(d->plane_gpu, plane, status);
            cufftExecZ2Z(h->cufft_plan, oskar_mem_void(d->plane_gpu),
                    oskar_mem_void(d->plane_gpu), CUFFT_FORWARD);
            oskar_mem_copy(plane, d->plane_gpu, status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
        {
            oskar_fftpack_cfft2f(size, size, size,
                    oskar_mem_double(plane, status),
                    oskar_mem_double(h->fftpack_wsave, status),
                    oskar_mem_double(h->fftpack_work, status));
            oskar_mem_scale_real(plane, (double)num_cells, status);
        }
        oskar_fftphase_cd(size, size, oskar_mem_double(plane, status));
        oskar_grid_correction_d(size, oskar_mem_double(h->corr_func, status),
                oskar_mem_double(plane, status));
    }
    else
    {
        oskar_fftphase_cf(size, size, oskar_mem_float(plane, status));
        if (h->fft_on_gpu)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_device_set(h->cuda_device_ids[0], status);
            oskar_mem_copy(d->plane_gpu, plane, status);
            cufftExecC2C(h->cufft_plan, oskar_mem_void(d->plane_gpu),
                    oskar_mem_void(d->plane_gpu), CUFFT_FORWARD);
            oskar_mem_copy(plane, d->plane_gpu, status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
        {
            oskar_fftpack_cfft2f_f(size, size, size,
                    oskar_mem_float(plane, status),
                    oskar_mem_float(h->fftpack_wsave, status),
                    oskar_mem_float(h->fftpack_work, status));
            oskar_mem_scale_real(plane, (double)num_cells, status);
        }
        oskar_fftphase_cf(size, size, oskar_mem_float(plane, status));
        oskar_grid_correction_f(size, oskar_mem_double(h->corr_func, status),
                oskar_mem_float(plane, status));
    }
}


void oskar_imager_trim_image(oskar_Mem* plane,
        int plane_size, int image_size, int* status)
{
    int i, num_cells, size_diff;
    if (*status) return;

    /* Get the real part only, if the plane is complex. */
    if (oskar_mem_is_complex(plane))
    {
        num_cells = plane_size * plane_size;
        if (oskar_mem_precision(plane) == OSKAR_DOUBLE)
        {
            double *t = oskar_mem_double(plane, status);
            for (i = 0; i < num_cells; ++i) t[i] = t[2 * i];
        }
        else
        {
            float *t = oskar_mem_float(plane, status);
            for (i = 0; i < num_cells; ++i) t[i] = t[2 * i];
        }
    }

    /* Trim to required image size. */
    size_diff = plane_size - image_size;
    if (size_diff > 0)
    {
        char *ptr;
        size_t in = 0, out = 0, copy_len = 0, element_size = 0;
        ptr = oskar_mem_char(plane);
        element_size = oskar_mem_element_size(oskar_mem_precision(plane));
        copy_len = element_size * image_size;
        in = element_size * (size_diff / 2) * (plane_size + 1);
        for (i = 0; i < image_size; ++i)
        {
            /* Use memmove() instead of memcpy() to allow for overlap. */
            memmove(ptr + out, ptr + in, copy_len);
            in += plane_size * element_size;
            out += copy_len;
        }
    }
}


void write_plane(oskar_Imager* h, oskar_Mem* plane,
        int t, int c, int p, int* status)
{
    int datatype, num_pixels;
    long firstpix[4];
    if (*status) return;
    if (!h->fits_file[p]) return;
    datatype = (oskar_mem_is_double(plane) ? TDOUBLE : TFLOAT);
    firstpix[0] = 1;
    firstpix[1] = 1;
    firstpix[2] = 1 + c;
    firstpix[3] = 1 + t;
    num_pixels = h->image_size * h->image_size;
    fits_write_pix(h->fits_file[p], datatype, firstpix, num_pixels,
            oskar_mem_void(plane), status);
}


#ifdef __cplusplus
}
#endif
