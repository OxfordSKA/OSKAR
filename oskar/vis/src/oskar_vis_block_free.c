/*
 * Copyright (c) 2015-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "vis/private_vis_block.h"
#include "vis/oskar_vis_block.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_vis_block_free(oskar_VisBlock* vis, int* status)
{
    int i;
    if (!vis) return;
    for (i = 0; i < 3; ++i)
    {
        oskar_mem_free(vis->baseline_uvw_metres[i], status);
        oskar_mem_free(vis->station_uvw_metres[i], status);
    }
    oskar_mem_free(vis->auto_correlations, status);
    oskar_mem_free(vis->cross_correlations, status);
    free(vis);
}

#ifdef __cplusplus
}
#endif
