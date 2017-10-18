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

#include <stdlib.h>

#include "imager/private_imager.h"
#include "imager/private_imager_update_plane_dft.h"
#include "imager/oskar_imager.h"
#include "math/oskar_cmath.h"
#include "math/oskar_dft_c2r.h"
#include "utility/oskar_device_utils.h"
#include "utility/oskar_thread.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

static void* run_blocks(void* arg);

struct ThreadArgs
{
    oskar_Imager* h;
    oskar_Mem* plane;
    int thread_id, num_vis;
};
typedef struct ThreadArgs ThreadArgs;

void oskar_imager_update_plane_dft(oskar_Imager* h, size_t num_vis,
        const oskar_Mem* uu, const oskar_Mem* vv, const oskar_Mem* ww,
        const oskar_Mem* amps, const oskar_Mem* weight, oskar_Mem* plane,
        double* plane_norm, int* status)
{
    size_t i, num_pixels, num_threads;
    oskar_Thread** threads = 0;
    ThreadArgs* args = 0;
    if (*status) return;

    /* Check the image plane. */
    num_pixels = (size_t) h->image_size;
    num_pixels *= num_pixels;
    if (oskar_mem_precision(plane) != h->imager_prec)
        *status = OSKAR_ERR_TYPE_MISMATCH;
    if (oskar_mem_is_complex(plane) || oskar_mem_is_matrix(plane))
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    if (oskar_mem_length(plane) < num_pixels)
        oskar_mem_realloc(plane, num_pixels, status);
    if (*status) return;

    /* Copy visibility data to each device. */
    num_threads = (size_t) (h->num_devices);
    for (i = 0; i < num_threads; ++i)
    {
        if (i < (size_t) (h->num_gpus))
            oskar_device_set(h->gpu_ids[i], status);
        oskar_mem_copy(h->d[i].uu, uu, status);
        oskar_mem_copy(h->d[i].vv, vv, status);
        oskar_mem_copy(h->d[i].amp, amps, status);
        oskar_mem_copy(h->d[i].weight, weight, status);
        if (h->algorithm == OSKAR_ALGORITHM_DFT_3D)
            oskar_mem_copy(h->d[i].ww, ww, status);
    }

    /* Set up worker threads. */
    threads = (oskar_Thread**) calloc(num_threads, sizeof(oskar_Thread*));
    args = (ThreadArgs*) calloc(num_threads, sizeof(ThreadArgs));
    for (i = 0; i < num_threads; ++i)
    {
        args[i].h = h;
        args[i].thread_id = (int) i;
        args[i].num_vis = (int) num_vis;
        args[i].plane = plane;
    }

    /* Set status code. */
    h->status = *status;

    /* Start the worker threads. */
    h->i_block = 0;
    for (i = 0; i < num_threads; ++i)
        threads[i] = oskar_thread_create(run_blocks, (void*)&args[i], 0);

    /* Wait for worker threads to finish. */
    for (i = 0; i < num_threads; ++i)
    {
        oskar_thread_join(threads[i]);
        oskar_thread_free(threads[i]);
    }
    free(threads);
    free(args);

    /* Get status code. */
    *status = h->status;

    /* Update normalisation. */
    if (oskar_mem_precision(weight) == OSKAR_DOUBLE)
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
}

static void* run_blocks(void* arg)
{
    oskar_Imager* h;
    oskar_Mem *t, *plane;
    DeviceData* d;
    size_t max_block_size, num_pixels;
    const size_t smallest = 1024, largest = 65536;
    int i_block, thread_id, num_blocks, num_vis;
    int *status;

    /* Get thread function arguments. */
    h = ((ThreadArgs*)arg)->h;
    thread_id = ((ThreadArgs*)arg)->thread_id;
    num_vis = ((ThreadArgs*)arg)->num_vis;
    plane = ((ThreadArgs*)arg)->plane;
    status = &(h->status);

    /* Set the device used by the thread. */
    d = &h->d[thread_id];
    if (thread_id < h->num_gpus)
        oskar_device_set(h->gpu_ids[thread_id], status);

#ifdef _OPENMP
    /* Disable nested parallelism. */
    omp_set_nested(0);
    omp_set_num_threads(1);
#endif

    /* Pointer to output block. */
    t = oskar_mem_create_alias(0, 0, 0, status);

    /* Calculate the maximum pixel block size, and number of blocks. */
    num_pixels = h->image_size * h->image_size;
    max_block_size = num_pixels / h->num_devices;
    max_block_size = ((max_block_size + smallest - 1) / smallest) * smallest;
    if (max_block_size > largest) max_block_size = largest;
    if (max_block_size < smallest) max_block_size = smallest;
    num_blocks = (int) ((num_pixels + max_block_size - 1) / max_block_size);

    /* Loop until all blocks are done. */
    for (;;)
    {
        size_t block_size, block_start;

        /* Get a unique block index. */
        oskar_mutex_lock(h->mutex);
        i_block = (h->i_block)++;
        oskar_mutex_unlock(h->mutex);
        if ((i_block >= num_blocks) || *status) break;

        /* Calculate the block size. */
        block_start = i_block * max_block_size;
        block_size = num_pixels - block_start;
        if (block_size > max_block_size) block_size = max_block_size;

        /* Copy the l,m,n positions for the block. */
        if (oskar_mem_length(d->l) < block_size)
            oskar_mem_realloc(d->l, block_size, status);
        oskar_mem_copy_contents(d->l, h->l, 0, block_start,
                block_size, status);
        if (oskar_mem_length(d->m) < block_size)
            oskar_mem_realloc(d->m, block_size, status);
        oskar_mem_copy_contents(d->m, h->m, 0, block_start,
                block_size, status);
        if (h->algorithm == OSKAR_ALGORITHM_DFT_3D)
        {
            if (oskar_mem_length(d->n) < block_size)
                oskar_mem_realloc(d->n, block_size, status);
            oskar_mem_copy_contents(d->n, h->n, 0, block_start,
                    block_size, status);
        }

        /* Run DFT for the block. */
        oskar_dft_c2r(num_vis, 2.0 * M_PI, d->uu, d->vv, d->ww,
                d->amp, d->weight, (int) block_size,
                d->l, d->m, d->n, d->block_dev, status);

        /* Copy data to the host and add to existing pixels. */
        oskar_mem_copy(d->block_cpu, d->block_dev, status);
        oskar_mem_set_alias(t, plane, block_start, block_size, status);
        oskar_mem_add(t, t, d->block_cpu, block_size, status);
    }
    oskar_mem_free(t, status);
    return 0;
}

#ifdef __cplusplus
}
#endif
