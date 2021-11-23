/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>

#include "imager/private_imager.h"
#include "imager/private_imager_update_plane_dft.h"
#include "imager/oskar_imager.h"
#include "math/oskar_cmath.h"
#include "math/oskar_dft_c2r.h"
#include "utility/oskar_device.h"
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
    oskar_Mem *plane;
    const oskar_Mem *uu, *vv, *ww, *amp, *weight;
    int thread_id, num_vis;
};
typedef struct ThreadArgs ThreadArgs;

void oskar_imager_update_plane_dft(oskar_Imager* h, size_t num_vis,
        const oskar_Mem* uu, const oskar_Mem* vv, const oskar_Mem* ww,
        const oskar_Mem* amps, const oskar_Mem* weight, int i_plane,
        oskar_Mem* plane, double* plane_norm, int* status)
{
    size_t i = 0, num_pixels = 0;
    oskar_Mem* plane_ptr = 0;
    oskar_Thread** threads = 0;
    ThreadArgs* args = 0;
    if (*status) return;

    /* Check the image plane. */
    plane_ptr = plane;
    if (!plane_ptr)
    {
        if (h->planes)
        {
            plane_ptr = h->planes[i_plane];
        }
        else
        {
            *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
            return;
        }
    }
    num_pixels = (size_t) h->image_size;
    num_pixels *= num_pixels;
    if (oskar_mem_precision(plane_ptr) != h->imager_prec)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
    }
    if (oskar_mem_is_complex(plane_ptr) || oskar_mem_is_matrix(plane_ptr))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    oskar_mem_ensure(plane_ptr, num_pixels, status);
    if (*status) return;

    /* Set up worker threads. */
    const size_t num_threads = (size_t) (h->num_devices);
    threads = (oskar_Thread**) calloc(num_threads, sizeof(oskar_Thread*));
    args = (ThreadArgs*) calloc(num_threads, sizeof(ThreadArgs));
    for (i = 0; i < num_threads; ++i)
    {
        args[i].h = h;
        args[i].thread_id = (int) i;
        args[i].num_vis = (int) num_vis;
        args[i].uu = uu;
        args[i].vv = vv;
        args[i].ww = ww;
        args[i].amp = amps;
        args[i].weight = weight;
        args[i].plane = plane_ptr;
    }

    /* Set status code. */
    h->status = *status;

    /* Start the worker threads. */
    h->i_block = 0;
    for (i = 0; i < num_threads; ++i)
    {
        threads[i] = oskar_thread_create(run_blocks, (void*)&args[i], 0);
    }

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
        const double* w = oskar_mem_double_const(weight, status);
        for (i = 0; i < num_vis; ++i) *plane_norm += w[i];
    }
    else
    {
        const float* w = oskar_mem_float_const(weight, status);
        for (i = 0; i < num_vis; ++i) *plane_norm += w[i];
    }
}

static void* run_blocks(void* arg)
{
    oskar_Imager* h = 0;
    oskar_Mem *plane = 0, *uu = 0, *vv = 0, *ww = 0, *amp = 0, *weight = 0;
    oskar_Mem *block = 0, *l = 0, *m = 0, *n = 0;
    size_t max_size = 0;
    const size_t smallest = 1024, largest = 65536;
    int dev_loc = OSKAR_CPU, *status = 0;

    /* Get thread function arguments. */
    h = ((ThreadArgs*)arg)->h;
    const int thread_id = ((ThreadArgs*)arg)->thread_id;
    const int num_vis = ((ThreadArgs*)arg)->num_vis;
    plane = ((ThreadArgs*)arg)->plane;
    status = &(h->status);

    /* Set the device used by the thread. */
    if (thread_id < h->num_gpus)
    {
        dev_loc = h->dev_loc;
        oskar_device_set(h->dev_loc, h->gpu_ids[thread_id], status);
    }

    /* Copy visibility data to device. */
    uu = oskar_mem_create_copy(((ThreadArgs*)arg)->uu, dev_loc, status);
    vv = oskar_mem_create_copy(((ThreadArgs*)arg)->vv, dev_loc, status);
    amp = oskar_mem_create_copy(((ThreadArgs*)arg)->amp, dev_loc, status);
    weight = oskar_mem_create_copy(((ThreadArgs*)arg)->weight, dev_loc, status);
    if (h->algorithm == OSKAR_ALGORITHM_DFT_3D)
    {
        ww = oskar_mem_create_copy(((ThreadArgs*)arg)->ww, dev_loc, status);
    }

#ifdef _OPENMP
    /* Disable nested parallelism. */
    omp_set_nested(0);
    omp_set_num_threads(1);
#endif

    /* Calculate the maximum pixel block size, and number of blocks. */
    const size_t num_pixels = (size_t)h->image_size * (size_t)h->image_size;
    max_size = num_pixels / h->num_devices;
    max_size = ((max_size + smallest - 1) / smallest) * smallest;
    if (max_size > largest) max_size = largest;
    if (max_size < smallest) max_size = smallest;
    const int num_blocks = (int) ((num_pixels + max_size - 1) / max_size);

    /* Allocate device memory for pixel block data. */
    block = oskar_mem_create(h->imager_prec, dev_loc, 0, status);
    l = oskar_mem_create(h->imager_prec, dev_loc, max_size, status);
    m = oskar_mem_create(h->imager_prec, dev_loc, max_size, status);
    n = oskar_mem_create(h->imager_prec, dev_loc, max_size, status);

    /* Loop until all blocks are done. */
    for (;;)
    {
        size_t block_size = 0;

        /* Get a unique block index. */
        oskar_mutex_lock(h->mutex);
        const int i_block = (h->i_block)++;
        oskar_mutex_unlock(h->mutex);
        if ((i_block >= num_blocks) || *status) break;

        /* Calculate the block size. */
        const size_t block_start = i_block * max_size;
        block_size = num_pixels - block_start;
        if (block_size > max_size) block_size = max_size;

        /* Copy the (l,m,n) positions for the block. */
        oskar_mem_copy_contents(l, h->l, 0, block_start, block_size, status);
        oskar_mem_copy_contents(m, h->m, 0, block_start, block_size, status);
        if (h->algorithm == OSKAR_ALGORITHM_DFT_3D)
        {
            oskar_mem_copy_contents(n, h->n, 0, block_start,
                    block_size, status);
        }

        /* Run DFT for the block. */
        oskar_dft_c2r(num_vis, 2.0 * M_PI, uu, vv, ww, amp, weight,
                (int) block_size, l, m, n, block, status);

        /* Add data to existing pixels. */
        oskar_mem_add(plane, plane, block,
                block_start, block_start, 0, block_size, status);
    }

    /* Free memory. */
    oskar_mem_free(uu, status);
    oskar_mem_free(vv, status);
    oskar_mem_free(ww, status);
    oskar_mem_free(amp, status);
    oskar_mem_free(weight, status);
    oskar_mem_free(block, status);
    oskar_mem_free(l, status);
    oskar_mem_free(m, status);
    oskar_mem_free(n, status);
    return 0;
}

#ifdef __cplusplus
}
#endif
