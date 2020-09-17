/*
 * Copyright (c) 2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_station_uvw_to_baseline_uvw.h"
#include "vis/private_vis_block.h"
#include "vis/oskar_vis_block.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_vis_block_station_to_baseline_coords(oskar_VisBlock* vis,
        int* status)
{
    int i;
    if (*status) return;

    /* Convert station (u,v,w) to baseline (u,v,w). */
    const int num_times = oskar_vis_block_num_times(vis);
    const int num_stations = oskar_vis_block_num_stations(vis);
    const int num_baselines = oskar_vis_block_num_baselines(vis);
    for (i = 0; i < num_times; ++i)
    {
        oskar_convert_station_uvw_to_baseline_uvw(
                num_stations,
                num_stations * i,
                oskar_vis_block_station_uvw_metres(vis, 0),
                oskar_vis_block_station_uvw_metres(vis, 1),
                oskar_vis_block_station_uvw_metres(vis, 2),
                num_baselines * i,
                oskar_vis_block_baseline_uvw_metres(vis, 0),
                oskar_vis_block_baseline_uvw_metres(vis, 1),
                oskar_vis_block_baseline_uvw_metres(vis, 2),
                status);
    }
}

#ifdef __cplusplus
}
#endif
