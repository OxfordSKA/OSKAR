/*
 * Copyright (c) 2015-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "vis/private_vis_block.h"
#include "vis/oskar_vis_block.h"
#include "binary/oskar_binary.h"
#include "mem/oskar_binary_write_mem.h"


#ifdef __cplusplus
extern "C" {
#endif

void oskar_vis_block_write(const oskar_VisBlock* vis, oskar_Binary* h,
        int block_index, int* status)
{
    if (*status) return;

    /* Write visibility metadata. */
    oskar_binary_write(h, OSKAR_INT,
            OSKAR_TAG_GROUP_VIS_BLOCK,
            OSKAR_VIS_BLOCK_TAG_DIM_START_AND_SIZE, block_index,
            sizeof(int) * 6, vis->dim_start_size, status);

    /* Write the auto-correlation data. */
    if (oskar_vis_block_has_auto_correlations(vis))
    {
        oskar_binary_write_mem(h, vis->auto_correlations,
                OSKAR_TAG_GROUP_VIS_BLOCK,
                OSKAR_VIS_BLOCK_TAG_AUTO_CORRELATIONS, block_index, 0, status);
    }

    /* Write the cross-correlation data. */
    if (oskar_vis_block_has_cross_correlations(vis))
    {
        oskar_binary_write_mem(h, vis->cross_correlations,
                OSKAR_TAG_GROUP_VIS_BLOCK,
                OSKAR_VIS_BLOCK_TAG_CROSS_CORRELATIONS, block_index, 0, status);

        /* Write the station coordinate data. */
        oskar_binary_write_mem(h, vis->station_uvw_metres[0],
                OSKAR_TAG_GROUP_VIS_BLOCK,
                OSKAR_VIS_BLOCK_TAG_STATION_U, block_index, 0, status);
        oskar_binary_write_mem(h, vis->station_uvw_metres[1],
                OSKAR_TAG_GROUP_VIS_BLOCK,
                OSKAR_VIS_BLOCK_TAG_STATION_V, block_index, 0, status);
        oskar_binary_write_mem(h, vis->station_uvw_metres[2],
                OSKAR_TAG_GROUP_VIS_BLOCK,
                OSKAR_VIS_BLOCK_TAG_STATION_W, block_index, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
