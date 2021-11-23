/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager.h"
#include "imager/oskar_imager.h"

#include "imager/define_grid_tile_grid.h"
#include "imager/private_imager_update_plane_fft.h"
#include "imager/oskar_grid_simple.h"
#include "math/oskar_prefix_sum.h"
#include "math/oskar_round_robin.h"
#include "utility/oskar_device.h"
#include "utility/oskar_thread.h"

#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

static void* run_subset(void* arg);

struct ThreadArgs
{
    oskar_Imager* h;
    size_t num_vis, num_skipped;
    const oskar_Mem *uu, *vv, *amps, *weight;
    oskar_Mem *plane;
    double plane_norm;
    int grid_size, i_plane, thread_id;
};
typedef struct ThreadArgs ThreadArgs;

void oskar_imager_update_plane_fft(oskar_Imager* h, size_t num_vis,
        const oskar_Mem* uu, const oskar_Mem* vv, const oskar_Mem* amps,
        const oskar_Mem* weight, int i_plane, oskar_Mem* plane,
        double* plane_norm, size_t* num_skipped, int* status)
{
    if (*status) return;
    if (!h->grid_on_gpu || h->num_gpus == 0)
    {
        oskar_Mem* plane_ptr = plane;
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
        if (oskar_mem_location(plane_ptr) != OSKAR_CPU)
        {
            *status = OSKAR_ERR_LOCATION_MISMATCH;
            return;
        }
        if (oskar_mem_precision(plane_ptr) != h->imager_prec)
        {
            *status = OSKAR_ERR_TYPE_MISMATCH;
            return;
        }
        const int grid_size = oskar_imager_plane_size(h);
        const size_t num_cells = ((size_t) grid_size) * ((size_t) grid_size);
        oskar_mem_ensure(plane_ptr, num_cells, status);
        if (*status) return;
        if (h->imager_prec == OSKAR_DOUBLE)
        {
            oskar_grid_simple_d(h->support, h->oversample,
                    oskar_mem_double_const(h->conv_func, status), num_vis,
                    oskar_mem_double_const(uu, status),
                    oskar_mem_double_const(vv, status),
                    oskar_mem_double_const(amps, status),
                    oskar_mem_double_const(weight, status),
                    h->cellsize_rad,
                    grid_size, num_skipped, plane_norm,
                    oskar_mem_double(plane_ptr, status));
        }
        else
        {
            oskar_grid_simple_f(h->support, h->oversample,
                    oskar_mem_float_const(h->conv_func, status), num_vis,
                    oskar_mem_float_const(uu, status),
                    oskar_mem_float_const(vv, status),
                    oskar_mem_float_const(amps, status),
                    oskar_mem_float_const(weight, status),
                    (float) (h->cellsize_rad),
                    grid_size, num_skipped, plane_norm,
                    oskar_mem_float(plane_ptr, status));
        }
    }
    else
    {
        int i = 0;
        oskar_Thread** threads = 0;
        ThreadArgs* args = 0;

        /* Set up worker threads. */
        const int num_threads = h->num_gpus;
        threads = (oskar_Thread**) calloc(num_threads, sizeof(oskar_Thread*));
        args = (ThreadArgs*) calloc(num_threads, sizeof(ThreadArgs));
        for (i = 0; i < num_threads; ++i)
        {
            args[i].h = h;
            args[i].num_vis = num_vis;
            args[i].uu = uu;
            args[i].vv = vv;
            args[i].amps = amps;
            args[i].weight = weight;
            args[i].plane = plane;
            args[i].i_plane = i_plane;
            args[i].thread_id = i;
        }

        /* Set status code. */
        h->status = *status;

        /* Start the worker threads. */
        for (i = 0; i < num_threads; ++i)
        {
            threads[i] = oskar_thread_create(run_subset, (void*)&args[i], 0);
        }

        /* Wait for worker threads to finish. */
        for (i = 0; i < num_threads; ++i)
        {
            oskar_thread_join(threads[i]);
            oskar_thread_free(threads[i]);
            *plane_norm += args[i].plane_norm;
            *num_skipped += args[i].num_skipped;
        }
        free(threads);
        free(args);

        /* Get status code. */
        *status = h->status;
    }
}

static void* run_subset(void* arg)
{
    oskar_Imager* h = 0;
    oskar_Mem *plane = 0;
    const oskar_Mem *uu = 0, *vv = 0, *amps = 0, *weight = 0;
    int count_skipped = 0, num_total = 0, *status = 0;
    int start = 0, num_points = 0;
    size_t local_size[] = {1, 1, 1}, global_size[] = {1, 1, 1};
    DeviceData* d = 0;

    /* Get thread function arguments. */
    ThreadArgs* a = (ThreadArgs*) arg;
    const int i_plane = a->i_plane;
    const int thread_id = a->thread_id;
    const size_t num_vis = a->num_vis;
    h = a->h;
    status = &(h->status);
    uu = a->uu;
    vv = a->vv;
    amps = a->amps;
    weight = a->weight;

    /* Set the device used by the thread. */
    d = &h->d[thread_id];
    plane = a->plane;
    if (!plane)
    {
        if (d->planes)
        {
            plane = d->planes[i_plane];
        }
        else
        {
            *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
            return 0;
        }
    }
    if (oskar_mem_location(plane) != h->dev_loc)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return 0;
    }
    if (oskar_mem_precision(plane) != h->imager_prec)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return 0;
    }
    oskar_device_set(h->dev_loc, h->gpu_ids[thread_id], status);

    const int location = h->dev_loc;
    const int grid_size = oskar_imager_plane_size(h);
    const int vis_type = oskar_mem_type(amps);
    const int is_dbl = oskar_type_is_double(vis_type);
    const int grid_centre = grid_size / 2;
    const double grid_scale = grid_size * h->cellsize_rad;
    const float grid_scale_f = (float) grid_scale;

    /* Define the tile size and number of tiles in each direction.
     * A tile consists of SHMSZ grid cells per thread in shared memory
     * and REGSZ grid cells per thread in registers. */
    const int tile_size_u = 32;
    const int tile_size_v = (SHMSZ + REGSZ);
    const int num_tiles_u = (grid_size + tile_size_u - 1) / tile_size_u;
    const int num_tiles_v = (grid_size + tile_size_v - 1) / tile_size_v;
    const int num_tiles = num_tiles_u * num_tiles_v;

    /* Which tile contains the grid centre? */
    const int c_tile_u = grid_centre / tile_size_u;
    const int c_tile_v = grid_centre / tile_size_v;

    /* Compute difference between centre of centre tile and grid centre
     * to ensure the centre of the grid is in the centre of a tile. */
    const int top_left_u = grid_centre -
            c_tile_u * tile_size_u - tile_size_u / 2;
    const int top_left_v = grid_centre -
            c_tile_v * tile_size_v - tile_size_v / 2;
    assert(top_left_u <= 0);
    assert(top_left_v <= 0);

    /* Set up scratch memory. */
    oskar_round_robin((int)num_vis, h->num_gpus, thread_id,
            &num_points, &start);
    oskar_mem_ensure(d->uu, num_points, status);
    oskar_mem_ensure(d->vv, num_points, status);
    oskar_mem_ensure(d->vis, num_points, status);
    oskar_mem_ensure(d->weight, num_points, status);
    oskar_mem_copy_contents(d->uu, uu, 0, start, num_points, status);
    oskar_mem_copy_contents(d->vv, vv, 0, start, num_points, status);
    oskar_mem_copy_contents(d->vis, amps, 0, start, num_points, status);
    oskar_mem_copy_contents(d->weight, weight, 0, start, num_points, status);
    oskar_mem_ensure(d->num_points_in_tiles, num_tiles, status);
    oskar_mem_ensure(d->tile_offsets, num_tiles + 1, status);
    oskar_mem_ensure(d->tile_locks, num_tiles, status);
    oskar_mem_clear_contents(d->counter, status);
    oskar_mem_clear_contents(d->count_skipped, status);
    oskar_mem_clear_contents(d->norm, status);
    oskar_mem_clear_contents(d->num_points_in_tiles, status);
    oskar_mem_clear_contents(d->tile_locks, status);
    /* Don't need to clear d->tile_offsets, as it gets overwritten. */

    /* Count the number of elements in each tile. */
    const float inv_tile_size_u = 1.0f / (float) tile_size_u;
    const float inv_tile_size_v = 1.0f / (float) tile_size_v;
    {
        const char* k = 0;
        if (oskar_type_is_single(vis_type))
        {
            k = "grid_tile_count_simple_float";
        }
        else if (oskar_type_is_double(vis_type))
        {
            k = "grid_tile_count_simple_double";
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
        local_size[0] = 512;
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(num_points, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &h->support},
                {INT_SZ, &num_points},
                {PTR_SZ, oskar_mem_buffer_const(d->uu)},
                {PTR_SZ, oskar_mem_buffer_const(d->vv)},
                {INT_SZ, &grid_size},
                {INT_SZ, &grid_centre},
                {is_dbl ? DBL_SZ : FLT_SZ,  is_dbl ?
                        (const void*)&grid_scale :
                        (const void*)&grid_scale_f},
                {FLT_SZ, (const void*)&inv_tile_size_u},
                {FLT_SZ, (const void*)&inv_tile_size_v},
                {INT_SZ, &num_tiles_u},
                {INT_SZ, &top_left_u},
                {INT_SZ, &top_left_v},
                {PTR_SZ, oskar_mem_buffer(d->num_points_in_tiles)},
                {PTR_SZ, oskar_mem_buffer(d->count_skipped)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }

    /* Get the offsets for each tile using prefix sum. */
    oskar_prefix_sum(num_tiles,
            d->num_points_in_tiles, d->tile_offsets, status);

    /* Get the total number of visibilities to process. */
    oskar_mem_read_element(d->tile_offsets, num_tiles, &num_total, status);
    oskar_mem_read_element(d->count_skipped, 0, &count_skipped, status);
    a->num_skipped = (size_t) count_skipped;
    /*printf("Total points: %d (factor %.3f increase)\n", num_total,
            (float)num_total / (float)num_points);*/

    /* Bucket sort the data into tiles. */
    oskar_mem_ensure(d->sorted_uu, num_total, status);
    oskar_mem_ensure(d->sorted_vv, num_total, status);
    oskar_mem_ensure(d->sorted_wt, num_total, status);
    oskar_mem_ensure(d->sorted_vis, num_total, status);
    oskar_mem_ensure(d->sorted_tile, num_total, status);
    {
        const char* k = 0;
        if (oskar_type_is_single(vis_type))
        {
            k = "grid_tile_bucket_sort_simple_float";
        }
        else if (oskar_type_is_double(vis_type))
        {
            k = "grid_tile_bucket_sort_simple_double";
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
        local_size[0] = 128;
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(num_points, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &h->support},
                {INT_SZ, &num_points},
                {PTR_SZ, oskar_mem_buffer_const(d->uu)},
                {PTR_SZ, oskar_mem_buffer_const(d->vv)},
                {PTR_SZ, oskar_mem_buffer_const(d->vis)},
                {PTR_SZ, oskar_mem_buffer_const(d->weight)},
                {INT_SZ, &grid_size},
                {INT_SZ, &grid_centre},
                {is_dbl ? DBL_SZ : FLT_SZ,  is_dbl ?
                        (const void*)&grid_scale :
                        (const void*)&grid_scale_f},
                {FLT_SZ, (const void*)&inv_tile_size_u},
                {FLT_SZ, (const void*)&inv_tile_size_v},
                {INT_SZ, &num_tiles_u},
                {INT_SZ, &top_left_u},
                {INT_SZ, &top_left_v},
                {PTR_SZ, oskar_mem_buffer(d->tile_offsets)},
                {PTR_SZ, oskar_mem_buffer(d->sorted_uu)},
                {PTR_SZ, oskar_mem_buffer(d->sorted_vv)},
                {PTR_SZ, oskar_mem_buffer(d->sorted_vis)},
                {PTR_SZ, oskar_mem_buffer(d->sorted_wt)},
                {PTR_SZ, oskar_mem_buffer(d->sorted_tile)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }

    /* Update the grid. */
    {
        const char* k = 0;
        if (oskar_type_is_single(vis_type))
        {
            k = "grid_tile_grid_simple_float";
        }
        else if (oskar_type_is_double(vis_type))
        {
            k = "grid_tile_grid_simple_double";
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
        local_size[0] = tile_size_u;
        size_t num_blocks = (num_points + local_size[0] - 1) / local_size[0];
        if (num_blocks > 10000) num_blocks = 10000;
        global_size[0] = local_size[0] * num_blocks;
        const size_t sh_mem_size = oskar_mem_element_size(vis_type) *
                SHMSZ * local_size[0];
        const oskar_Arg args[] = {
                {INT_SZ, &h->support},
                {INT_SZ, &h->oversample},
                {PTR_SZ, oskar_mem_buffer_const(d->conv_func)},
                {INT_SZ, &grid_size},
                {INT_SZ, &grid_centre},
                {INT_SZ, &tile_size_u},
                {INT_SZ, &tile_size_v},
                {INT_SZ, &num_tiles_u},
                {INT_SZ, &top_left_u},
                {INT_SZ, &top_left_v},
                {INT_SZ, &num_total},
                {PTR_SZ, oskar_mem_buffer_const(d->sorted_uu)},
                {PTR_SZ, oskar_mem_buffer_const(d->sorted_vv)},
                {PTR_SZ, oskar_mem_buffer_const(d->sorted_vis)},
                {PTR_SZ, oskar_mem_buffer_const(d->sorted_wt)},
                {PTR_SZ, oskar_mem_buffer_const(d->sorted_tile)},
                {PTR_SZ, oskar_mem_buffer(d->tile_locks)},
                {PTR_SZ, oskar_mem_buffer(d->counter)},
                {PTR_SZ, oskar_mem_buffer(d->norm)},
                {PTR_SZ, oskar_mem_buffer(plane)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 1, &sh_mem_size,
                status);
    }

    /* Update the normalisation value. */
    if (oskar_mem_type(d->norm) == OSKAR_SINGLE)
    {
        float temp_norm = 0.0f;
        oskar_mem_read_element(d->norm, 0, &temp_norm, status);
        a->plane_norm = temp_norm;
    }
    else
    {
        double temp_norm = 0.0;
        oskar_mem_read_element(d->norm, 0, &temp_norm, status);
        a->plane_norm = temp_norm;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif
