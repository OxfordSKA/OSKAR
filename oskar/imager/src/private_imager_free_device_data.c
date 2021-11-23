/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager.h"
#include "imager/private_imager_free_device_data.h"
#include "utility/oskar_device.h"

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_free_device_scratch_data(oskar_Imager* h, int* status)
{
    int i = 0;
    for (i = 0; i < h->num_devices; ++i)
    {
        DeviceData* d = &(h->d[i]);
        if (!d) continue;
        if (i < h->num_gpus)
        {
            oskar_device_set(h->dev_loc, h->gpu_ids[i], status);
        }
        oskar_mem_free(d->uu, status);
        d->uu = 0;
        oskar_mem_free(d->vv, status);
        d->vv = 0;
        oskar_mem_free(d->ww, status);
        d->ww = 0;
        oskar_mem_free(d->vis, status);
        d->vis = 0;
        oskar_mem_free(d->weight, status);
        d->weight = 0;
        oskar_mem_free(d->counter, status);
        d->counter = 0;
        oskar_mem_free(d->count_skipped, status);
        d->count_skipped = 0;
        oskar_mem_free(d->norm, status);
        d->norm = 0;
        oskar_mem_free(d->num_points_in_tiles, status);
        d->num_points_in_tiles = 0;
        oskar_mem_free(d->tile_offsets, status);
        d->tile_offsets = 0;
        oskar_mem_free(d->tile_locks, status);
        d->tile_locks = 0;
        oskar_mem_free(d->sorted_uu, status);
        d->sorted_uu = 0;
        oskar_mem_free(d->sorted_vv, status);
        d->sorted_vv = 0;
        oskar_mem_free(d->sorted_ww, status);
        d->sorted_ww = 0;
        oskar_mem_free(d->sorted_wt, status);
        d->sorted_wt = 0;
        oskar_mem_free(d->sorted_vis, status);
        d->sorted_vis = 0;
        oskar_mem_free(d->sorted_tile, status);
        d->sorted_tile = 0;

        oskar_mem_free(d->conv_func, status);
        d->conv_func = 0;
        oskar_mem_free(d->w_support, status);
        d->w_support = 0;
        oskar_mem_free(d->w_kernels_compact, status);
        d->w_kernels_compact = 0;
        oskar_mem_free(d->w_kernel_start, status);
        d->w_kernel_start = 0;
    }
}

void oskar_imager_free_device_data(oskar_Imager* h, int* status)
{
    int i = 0, j = 0;
    oskar_imager_free_device_scratch_data(h, status);
    for (i = 0; i < h->num_devices; ++i)
    {
        DeviceData* d = &(h->d[i]);
        if (!d) continue;
        if (i < h->num_gpus)
        {
            oskar_device_set(h->dev_loc, h->gpu_ids[i], status);
        }
        for (j = 0; j < d->num_planes; ++j)
        {
            oskar_mem_free(d->planes[j], status);
        }
        free(d->planes);
        memset(d, 0, sizeof(DeviceData));
    }
}

#ifdef __cplusplus
}
#endif
