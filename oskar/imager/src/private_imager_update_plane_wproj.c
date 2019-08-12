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

#include "imager/define_grid_tile_grid.h"
#include "imager/private_imager_update_plane_wproj.h"
#include "imager/oskar_grid_wproj2.h"
#include "math/oskar_prefix_sum.h"
#include "utility/oskar_device.h"

#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_update_plane_wproj(oskar_Imager* h, size_t num_vis,
        const oskar_Mem* uu, const oskar_Mem* vv, const oskar_Mem* ww,
        const oskar_Mem* amps, const oskar_Mem* weight, oskar_Mem* plane,
        double* plane_norm, size_t* num_skipped, int* status)
{
    if (*status) return;
    const int grid_size = oskar_imager_plane_size(h);
    const int location = oskar_mem_location(plane);
    const int type = oskar_mem_precision(plane);
    const size_t num_cells = (size_t) grid_size * grid_size;
    if (type != h->imager_prec)
        *status = OSKAR_ERR_TYPE_MISMATCH;
    oskar_mem_ensure(plane, num_cells, status);
    if (*status) return;
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_DOUBLE)
            oskar_grid_wproj2_d(h->num_w_planes,
                    oskar_mem_int_const(h->w_support, status),
                    h->oversample,
                    oskar_mem_int_const(h->w_kernel_start, status),
                    oskar_mem_double_const(h->w_kernels_compact, status), num_vis,
                    oskar_mem_double_const(uu, status),
                    oskar_mem_double_const(vv, status),
                    oskar_mem_double_const(ww, status),
                    oskar_mem_double_const(amps, status),
                    oskar_mem_double_const(weight, status),
                    h->cellsize_rad, h->w_scale,
                    grid_size, num_skipped, plane_norm,
                    oskar_mem_double(plane, status));
        else
            oskar_grid_wproj2_f(h->num_w_planes,
                    oskar_mem_int_const(h->w_support, status),
                    h->oversample,
                    oskar_mem_int_const(h->w_kernel_start, status),
                    oskar_mem_float_const(h->w_kernels_compact, status), num_vis,
                    oskar_mem_float_const(uu, status),
                    oskar_mem_float_const(vv, status),
                    oskar_mem_float_const(ww, status),
                    oskar_mem_float_const(amps, status),
                    oskar_mem_float_const(weight, status),
                    h->cellsize_rad, h->w_scale,
                    grid_size, num_skipped, plane_norm,
                    oskar_mem_float(plane, status));
    }
    else
    {
        int count_skipped = 0, norm_type = 0, num_total = 0;
        size_t local_size[] = {1, 1, 1}, global_size[] = {1, 1, 1};
        oskar_Mem *d_uu, *d_vv, *d_ww, *d_vis, *d_weight;
        oskar_Mem *d_counter, *d_count_skipped, *d_norm;
        oskar_Mem *d_num_points_in_tiles, *d_tile_offsets, *d_tile_locks;
        oskar_Mem *sorted_uu, *sorted_vv, *sorted_ww;
        oskar_Mem *sorted_tile, *sorted_wt, *sorted_vis;
        DeviceData* d = &h->d[0];
        const int coord_type = oskar_mem_type(uu);
        const int vis_type = oskar_mem_type(amps);
        const int weight_type = oskar_mem_type(weight);
        const int is_dbl = oskar_type_is_double(vis_type);
        const int num_points = (int) num_vis;
        const int grid_centre = grid_size / 2;
        const double grid_scale = grid_size * h->cellsize_rad;
        const double w_scale = h->w_scale;
        const float grid_scale_f = (float) grid_scale;
        const float w_scale_f = (float) w_scale;

        /* Get the normalisation type. */
        if (oskar_device_supports_double(location) &&
                oskar_device_supports_atomic64(location))
            norm_type = OSKAR_DOUBLE;
        else
            norm_type = OSKAR_SINGLE;

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

        /* Allocate and set up scratch memory. */
        oskar_device_set(location, h->gpu_ids[0], status);
        d_uu = oskar_mem_create_copy(uu, location, status);
        d_vv = oskar_mem_create_copy(vv, location, status);
        d_ww = oskar_mem_create_copy(ww, location, status);
        d_vis = oskar_mem_create_copy(amps, location, status);
        d_weight = oskar_mem_create_copy(weight, location, status);
        d_counter = oskar_mem_create(OSKAR_INT, location, 1, status);
        d_count_skipped = oskar_mem_create(OSKAR_INT, location, 1, status);
        d_norm = oskar_mem_create(norm_type, location, 1, status);
        d_num_points_in_tiles = oskar_mem_create(OSKAR_INT,
                location, num_tiles, status);
        d_tile_offsets = oskar_mem_create(OSKAR_INT,
                location, num_tiles + 1, status);
        d_tile_locks = oskar_mem_create(OSKAR_INT, location, num_tiles, status);
        oskar_mem_clear_contents(d_counter, status);
        oskar_mem_clear_contents(d_count_skipped, status);
        oskar_mem_clear_contents(d_norm, status);
        oskar_mem_clear_contents(d_num_points_in_tiles, status);
        oskar_mem_clear_contents(d_tile_locks, status);
        /* Don't need to clear d_tile_offsets, as it gets overwritten. */

        /* Count the number of elements in each tile. */
        const float inv_tile_size_u = 1.0f / (float) tile_size_u;
        const float inv_tile_size_v = 1.0f / (float) tile_size_v;
        {
            const char* k = 0;
            if (oskar_type_is_single(vis_type))
                k = "grid_tile_count_wproj_float";
            else if (oskar_type_is_double(vis_type))
                k = "grid_tile_count_wproj_double";
            else
                *status = OSKAR_ERR_BAD_DATA_TYPE;
            local_size[0] = 512;
            oskar_device_check_local_size(location, 0, local_size);
            global_size[0] = oskar_device_global_size(num_points, local_size[0]);
            const oskar_Arg args[] = {
                    {INT_SZ, &h->num_w_planes},
                    {PTR_SZ, oskar_mem_buffer_const(d->w_support)},
                    {INT_SZ, &num_points},
                    {PTR_SZ, oskar_mem_buffer_const(d_uu)},
                    {PTR_SZ, oskar_mem_buffer_const(d_vv)},
                    {PTR_SZ, oskar_mem_buffer_const(d_ww)},
                    {INT_SZ, &grid_size},
                    {INT_SZ, &grid_centre},
                    {is_dbl ? DBL_SZ : FLT_SZ,  is_dbl ?
                            (const void*)&grid_scale :
                            (const void*)&grid_scale_f},
                    {is_dbl ? DBL_SZ : FLT_SZ,  is_dbl ?
                            (const void*)&w_scale :
                            (const void*)&w_scale_f},
                    {FLT_SZ, (const void*)&inv_tile_size_u},
                    {FLT_SZ, (const void*)&inv_tile_size_v},
                    {INT_SZ, &num_tiles_u},
                    {INT_SZ, &top_left_u},
                    {INT_SZ, &top_left_v},
                    {PTR_SZ, oskar_mem_buffer(d_num_points_in_tiles)},
                    {PTR_SZ, oskar_mem_buffer(d_count_skipped)}
            };
            oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                    sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
        }

        /* Get the offsets for each tile using prefix sum. */
        oskar_prefix_sum(num_tiles,
                d_num_points_in_tiles, d_tile_offsets, status);

        /* Get the total number of visibilities to process. */
        oskar_mem_read_element(d_tile_offsets, num_tiles, &num_total, status);
        oskar_mem_read_element(d_count_skipped, 0, &count_skipped, status);
        *num_skipped = (size_t) count_skipped;
        /*printf("Total points: %d (factor %.3f increase)\n", num_total,
                (float)num_total / (float)num_points);*/

        /* Bucket sort the data into tiles. */
        sorted_uu = oskar_mem_create(coord_type, location, num_total, status);
        sorted_vv = oskar_mem_create(coord_type, location, num_total, status);
        sorted_ww = oskar_mem_create(OSKAR_INT, location, num_total, status);
        sorted_wt = oskar_mem_create(weight_type, location, num_total, status);
        sorted_vis = oskar_mem_create(vis_type, location, num_total, status);
        sorted_tile = oskar_mem_create(OSKAR_INT, location, num_total, status);
        {
            const char* k = 0;
            if (oskar_type_is_single(vis_type))
                k = "grid_tile_bucket_sort_wproj_float";
            else if (oskar_type_is_double(vis_type))
                k = "grid_tile_bucket_sort_wproj_double";
            else
                *status = OSKAR_ERR_BAD_DATA_TYPE;
            local_size[0] = 128;
            oskar_device_check_local_size(location, 0, local_size);
            global_size[0] = oskar_device_global_size(num_vis, local_size[0]);
            const oskar_Arg args[] = {
                    {INT_SZ, &h->num_w_planes},
                    {PTR_SZ, oskar_mem_buffer_const(d->w_support)},
                    {INT_SZ, &num_points},
                    {PTR_SZ, oskar_mem_buffer_const(d_uu)},
                    {PTR_SZ, oskar_mem_buffer_const(d_vv)},
                    {PTR_SZ, oskar_mem_buffer_const(d_ww)},
                    {PTR_SZ, oskar_mem_buffer_const(d_vis)},
                    {PTR_SZ, oskar_mem_buffer_const(d_weight)},
                    {INT_SZ, &grid_size},
                    {INT_SZ, &grid_centre},
                    {is_dbl ? DBL_SZ : FLT_SZ,  is_dbl ?
                            (const void*)&grid_scale :
                            (const void*)&grid_scale_f},
                    {is_dbl ? DBL_SZ : FLT_SZ,  is_dbl ?
                            (const void*)&w_scale :
                            (const void*)&w_scale_f},
                    {FLT_SZ, (const void*)&inv_tile_size_u},
                    {FLT_SZ, (const void*)&inv_tile_size_v},
                    {INT_SZ, &num_tiles_u},
                    {INT_SZ, &top_left_u},
                    {INT_SZ, &top_left_v},
                    {PTR_SZ, oskar_mem_buffer(d_tile_offsets)},
                    {PTR_SZ, oskar_mem_buffer(sorted_uu)},
                    {PTR_SZ, oskar_mem_buffer(sorted_vv)},
                    {PTR_SZ, oskar_mem_buffer(sorted_ww)},
                    {PTR_SZ, oskar_mem_buffer(sorted_vis)},
                    {PTR_SZ, oskar_mem_buffer(sorted_wt)},
                    {PTR_SZ, oskar_mem_buffer(sorted_tile)}
            };
            oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                    sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
        }

        /* Update the grid. */
        {
            const char* k = 0;
            if (oskar_type_is_single(vis_type))
                k = "grid_tile_grid_wproj_float";
            else if (oskar_type_is_double(vis_type))
                k = "grid_tile_grid_wproj_double";
            else
                *status = OSKAR_ERR_BAD_DATA_TYPE;
            local_size[0] = tile_size_u;
            size_t num_blocks = (num_vis + local_size[0] - 1) / local_size[0];
            if (num_blocks > 10000) num_blocks = 10000;
            global_size[0] = local_size[0] * num_blocks;
            const size_t sh_mem_size = oskar_mem_element_size(vis_type) *
                    SHMSZ * local_size[0];
            const oskar_Arg args[] = {
                    {INT_SZ, &h->num_w_planes},
                    {PTR_SZ, oskar_mem_buffer_const(d->w_support)},
                    {INT_SZ, &h->oversample},
                    {PTR_SZ, oskar_mem_buffer_const(d->w_kernel_start)},
                    {PTR_SZ, oskar_mem_buffer_const(d->w_kernels_compact)},
                    {INT_SZ, &grid_size},
                    {INT_SZ, &grid_centre},
                    {INT_SZ, &tile_size_u},
                    {INT_SZ, &tile_size_v},
                    {INT_SZ, &num_tiles_u},
                    {INT_SZ, &top_left_u},
                    {INT_SZ, &top_left_v},
                    {INT_SZ, &num_total},
                    {PTR_SZ, oskar_mem_buffer_const(sorted_uu)},
                    {PTR_SZ, oskar_mem_buffer_const(sorted_vv)},
                    {PTR_SZ, oskar_mem_buffer_const(sorted_ww)},
                    {PTR_SZ, oskar_mem_buffer_const(sorted_vis)},
                    {PTR_SZ, oskar_mem_buffer_const(sorted_wt)},
                    {PTR_SZ, oskar_mem_buffer_const(sorted_tile)},
                    {PTR_SZ, oskar_mem_buffer(d_tile_locks)},
                    {PTR_SZ, oskar_mem_buffer(d_counter)},
                    {PTR_SZ, oskar_mem_buffer(d_norm)},
                    {PTR_SZ, oskar_mem_buffer(plane)}
            };
            oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                    sizeof(args) / sizeof(oskar_Arg), args, 1, &sh_mem_size,
                    status);
        }

        /* Update the normalisation value. */
        if (norm_type == OSKAR_SINGLE)
        {
            float temp_norm;
            oskar_mem_read_element(d_norm, 0, &temp_norm, status);
            *plane_norm += temp_norm;
        }
        else
        {
            double temp_norm;
            oskar_mem_read_element(d_norm, 0, &temp_norm, status);
            *plane_norm += temp_norm;
        }

        /* Free scratch memory. */
        oskar_mem_free(d_counter, status);
        oskar_mem_free(d_count_skipped, status);
        oskar_mem_free(d_norm, status);
        oskar_mem_free(d_num_points_in_tiles, status);
        oskar_mem_free(d_tile_offsets, status);
        oskar_mem_free(d_tile_locks, status);
        oskar_mem_free(d_uu, status);
        oskar_mem_free(d_vv, status);
        oskar_mem_free(d_ww, status);
        oskar_mem_free(d_vis, status);
        oskar_mem_free(d_weight, status);
        oskar_mem_free(sorted_uu, status);
        oskar_mem_free(sorted_vv, status);
        oskar_mem_free(sorted_ww, status);
        oskar_mem_free(sorted_wt, status);
        oskar_mem_free(sorted_vis, status);
        oskar_mem_free(sorted_tile, status);
    }
}

#ifdef __cplusplus
}
#endif
