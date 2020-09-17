/*
 * Copyright (c) 2016-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "vis/private_vis_block.h"
#include "vis/oskar_vis_block.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_vis_block_resize(oskar_VisBlock* vis, int num_times,
        int num_channels, int num_stations, int* status)
{
    int i = 0, num_autocorr = 0, num_baselines = 0;
    if (*status) return;

    /* Set dimensions. */
    if (vis->has_cross_correlations)
        num_baselines = num_stations * (num_stations - 1) / 2;
    vis->dim_start_size[2] = num_times;
    vis->dim_start_size[3] = num_channels;
    vis->dim_start_size[4] = num_baselines;
    vis->dim_start_size[5] = num_stations;
    const int num_xcorr = num_times * num_baselines * num_channels;
    if (vis->has_auto_correlations)
        num_autocorr = num_channels * num_times * num_stations;

    /* Resize arrays as required. */
    for (i = 0; i < 3; ++i)
    {
        if (oskar_mem_length(vis->baseline_uvw_metres[i]) > 0)
            oskar_mem_realloc(vis->baseline_uvw_metres[i],
                    num_times * num_baselines, status);
        if (oskar_mem_length(vis->station_uvw_metres[i]) > 0)
            oskar_mem_realloc(vis->station_uvw_metres[i],
                    num_times * num_stations, status);
    }
    oskar_mem_realloc(vis->auto_correlations, num_autocorr, status);
    oskar_mem_realloc(vis->cross_correlations, num_xcorr, status);
}

#ifdef __cplusplus
}
#endif
