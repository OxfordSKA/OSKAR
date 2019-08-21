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
#include "imager/private_imager_free_device_data.h"
#include "utility/oskar_device.h"

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_free_device_scratch_data(oskar_Imager* h, int* status)
{
    int i;
    for (i = 0; i < h->num_devices; ++i)
    {
        DeviceData* d = &(h->d[i]);
        if (!d) continue;
        if (i < h->num_gpus)
            oskar_device_set(h->dev_loc, h->gpu_ids[i], status);
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
    int i, j;
    oskar_imager_free_device_scratch_data(h, status);
    for (i = 0; i < h->num_devices; ++i)
    {
        DeviceData* d = &(h->d[i]);
        if (!d) continue;
        if (i < h->num_gpus)
            oskar_device_set(h->dev_loc, h->gpu_ids[i], status);
        for (j = 0; j < d->num_planes; ++j)
            oskar_mem_free(d->planes[j], status);
        free(d->planes);
        memset(d, 0, sizeof(DeviceData));
    }
}

#ifdef __cplusplus
}
#endif
