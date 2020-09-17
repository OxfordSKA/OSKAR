/*
 * Copyright (c) 2015-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "vis/private_vis_block.h"
#include "vis/oskar_vis_block.h"
#include "mem/oskar_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_vis_block_clear(oskar_VisBlock* vis, int* status)
{
    int i;
    if (*status) return;

    oskar_mem_clear_contents(vis->auto_correlations, status);
    oskar_mem_clear_contents(vis->cross_correlations, status);
    for (i = 0; i < 3; ++i)
    {
        oskar_mem_clear_contents(vis->baseline_uvw_metres[i], status);
        oskar_mem_clear_contents(vis->station_uvw_metres[i], status);
    }
}

#ifdef __cplusplus
}
#endif
