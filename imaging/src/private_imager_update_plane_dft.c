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

#include <oskar_dft_c2r_2d_cuda.h>
#include <oskar_dft_c2r_3d_cuda.h>
#include <oskar_device_utils.h>
#include <oskar_imager.h>
#include <oskar_cmath.h>
#include <private_imager_update_plane_dft.h>

#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_update_plane_dft(oskar_Imager* h, int num_vis,
        const oskar_Mem* uu, const oskar_Mem* vv, const oskar_Mem* ww,
        const oskar_Mem* amps, const oskar_Mem* weight, oskar_Mem* plane,
        double* plane_norm, int* status)
{
    int i, num_blocks, max_block_size = 65536, prec, num_pixels;
    oskar_Mem** t;
    if (*status) return;

    /* Check the image plane. */
    num_pixels = h->image_size * h->image_size;
    prec = oskar_mem_precision(amps);
    if (oskar_mem_precision(plane) != h->imager_prec)
        *status = OSKAR_ERR_TYPE_MISMATCH;
    if (oskar_mem_is_complex(plane) || oskar_mem_is_matrix(plane))
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    if ((int)oskar_mem_length(plane) < num_pixels)
        oskar_mem_realloc(plane, num_pixels, status);
    if (*status) return;

    /* Copy visibility data to each GPU. */
    t = (oskar_Mem**) calloc(h->num_gpus, sizeof(oskar_Mem*));
    for (i = 0; i < h->num_gpus; ++i)
    {
        oskar_device_set(h->cuda_device_ids[i], status);
        oskar_mem_copy(h->d[i].uu, uu, status);
        oskar_mem_copy(h->d[i].vv, vv, status);
        oskar_mem_copy(h->d[i].amp, amps, status);
        oskar_mem_copy(h->d[i].weight, weight, status);
        if (h->algorithm == OSKAR_ALGORITHM_DFT_3D)
            oskar_mem_copy(h->d[i].ww, ww, status);
        t[i] = oskar_mem_create_alias(0, 0, 0, status);
    }

#ifdef _OPENMP
    omp_set_num_threads(h->num_gpus);
#endif

    /* Loop over pixel blocks. */
    num_blocks = (num_pixels + max_block_size - 1) / max_block_size;
#pragma omp parallel for private(i)
    for (i = 0; i < num_blocks; ++i)
    {
        DeviceData* d;
        int thread_id = 0, block_size, block_start;
        if (*status) continue;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        d = &h->d[thread_id];
        oskar_device_set(h->cuda_device_ids[thread_id], status);

        /* Calculate the block size. */
        block_start = i * max_block_size;
        block_size = num_pixels - block_start;
        if (block_size > max_block_size) block_size = max_block_size;

        /* Ensure blocks are big enough. */
        if ((int)oskar_mem_length(d->l) < block_size)
            oskar_mem_realloc(d->l, block_size, status);
        if ((int)oskar_mem_length(d->m) < block_size)
            oskar_mem_realloc(d->m, block_size, status);
        if ((int)oskar_mem_length(d->block_gpu) < block_size)
            oskar_mem_realloc(d->block_gpu, block_size, status);

        /* Copy the l,m positions for the block. */
        oskar_mem_copy_contents(d->l, h->l, 0, block_start, block_size, status);
        oskar_mem_copy_contents(d->m, h->m, 0, block_start, block_size, status);

        if (h->algorithm == OSKAR_ALGORITHM_DFT_2D)
        {
            if (prec == OSKAR_DOUBLE)
                oskar_dft_c2r_2d_cuda_d(num_vis, 2.0 * M_PI,
                        oskar_mem_double_const(d->uu, status),
                        oskar_mem_double_const(d->vv, status),
                        oskar_mem_double2_const(d->amp, status),
                        oskar_mem_double_const(d->weight, status), block_size,
                        oskar_mem_double_const(d->l, status),
                        oskar_mem_double_const(d->m, status),
                        oskar_mem_double(d->block_gpu, status));
            else
                oskar_dft_c2r_2d_cuda_f(num_vis, 2.0 * M_PI,
                        oskar_mem_float_const(d->uu, status),
                        oskar_mem_float_const(d->vv, status),
                        oskar_mem_float2_const(d->amp, status),
                        oskar_mem_float_const(d->weight, status), block_size,
                        oskar_mem_float_const(d->l, status),
                        oskar_mem_float_const(d->m, status),
                        oskar_mem_float(d->block_gpu, status));
        }
        else if (h->algorithm == OSKAR_ALGORITHM_DFT_3D)
        {
            if ((int)oskar_mem_length(d->n) < block_size)
                oskar_mem_realloc(d->n, block_size, status);
            oskar_mem_copy_contents(d->n, h->n, 0, block_start,
                    block_size, status);
            if (prec == OSKAR_DOUBLE)
                oskar_dft_c2r_3d_cuda_d(num_vis, 2.0 * M_PI,
                        oskar_mem_double_const(d->uu, status),
                        oskar_mem_double_const(d->vv, status),
                        oskar_mem_double_const(d->ww, status),
                        oskar_mem_double2_const(d->amp, status),
                        oskar_mem_double_const(d->weight, status), block_size,
                        oskar_mem_double_const(d->l, status),
                        oskar_mem_double_const(d->m, status),
                        oskar_mem_double_const(d->n, status),
                        oskar_mem_double(d->block_gpu, status));
            else
                oskar_dft_c2r_3d_cuda_f(num_vis, 2.0 * M_PI,
                        oskar_mem_float_const(d->uu, status),
                        oskar_mem_float_const(d->vv, status),
                        oskar_mem_float_const(d->ww, status),
                        oskar_mem_float2_const(d->amp, status),
                        oskar_mem_float_const(d->weight, status), block_size,
                        oskar_mem_float_const(d->l, status),
                        oskar_mem_float_const(d->m, status),
                        oskar_mem_float_const(d->n, status),
                        oskar_mem_float(d->block_gpu, status));
        }
        oskar_device_check_error(status);

        /* Copy the data back and add to existing data. */
        oskar_mem_copy(d->block_cpu, d->block_gpu, status);
        oskar_mem_set_alias(t[thread_id], plane,
                block_start, block_size, status);
        oskar_mem_add(t[thread_id], t[thread_id], d->block_cpu,
                block_size, status);
    } /* End parallel loop over blocks. */

    /* Update normalisation. */
    if (prec == OSKAR_DOUBLE)
    {
        const double* w;
        w = oskar_mem_double_const(weight, status);
        for (i = 0; i < num_vis; ++i) *plane_norm += w[i];
    }
    else
    {
        const float* w;
        w = oskar_mem_float_const(weight, status);
        for (i = 0; i < num_vis; ++i) *plane_norm += w[i];
    }

    for (i = 0; i < h->num_gpus; ++i)
        oskar_mem_free(t[i], status);
    free(t);
}

#ifdef __cplusplus
}
#endif
