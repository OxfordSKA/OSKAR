/*
 * Copyright (c) 2015-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "binary/oskar_binary.h"
#include "mem/oskar_binary_read_mem.h"
#include "vis/private_vis_block.h"
#include "vis/oskar_vis_block.h"
#include "vis/oskar_vis_header.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_vis_block_read(oskar_VisBlock* vis, const oskar_VisHeader* hdr,
        oskar_Binary* h, int block_index, int* status)
{
    if (*status) return;

    /* Set query start index. */
    const int num_tags_per_block = oskar_vis_header_num_tags_per_block(hdr);
    oskar_binary_set_query_search_start(h, block_index * num_tags_per_block,
            status);

    /* Read visibility metadata. */
    oskar_binary_read(h, OSKAR_INT,
            OSKAR_TAG_GROUP_VIS_BLOCK,
            OSKAR_VIS_BLOCK_TAG_DIM_START_AND_SIZE, block_index,
            sizeof(int) * 6, vis->dim_start_size, status);

    /* Read the auto-correlation data. */
    if (oskar_vis_header_write_auto_correlations(hdr))
    {
        oskar_binary_read_mem(h, vis->auto_correlations,
                OSKAR_TAG_GROUP_VIS_BLOCK,
                OSKAR_VIS_BLOCK_TAG_AUTO_CORRELATIONS, block_index, status);
    }

    /* Read the cross-correlation data. */
    if (oskar_vis_header_write_cross_correlations(hdr))
    {
        int tag_error = 0;
        oskar_binary_read_mem(h, vis->cross_correlations,
                OSKAR_TAG_GROUP_VIS_BLOCK,
                OSKAR_VIS_BLOCK_TAG_CROSS_CORRELATIONS, block_index, status);

        /*
         * Read the station or baseline coordinate data.
         * Older files contained the baseline coordinates for the block,
         * which wasn't efficient use of storage.
         * Try to load the station coordinates and calculate the
         * baseline coordinates from those, otherwise read the
         * baseline coordinates directly.
         */
        oskar_binary_read_mem(h, vis->station_uvw_metres[0],
                OSKAR_TAG_GROUP_VIS_BLOCK,
                OSKAR_VIS_BLOCK_TAG_STATION_U, block_index, &tag_error);
        if (!tag_error)
        {
            oskar_binary_read_mem(h, vis->station_uvw_metres[1],
                    OSKAR_TAG_GROUP_VIS_BLOCK,
                    OSKAR_VIS_BLOCK_TAG_STATION_V, block_index, status);
            oskar_binary_read_mem(h, vis->station_uvw_metres[2],
                    OSKAR_TAG_GROUP_VIS_BLOCK,
                    OSKAR_VIS_BLOCK_TAG_STATION_W, block_index, status);
            oskar_vis_block_station_to_baseline_coords(vis, status);
        }
        else
        {
            oskar_binary_read_mem(h, vis->baseline_uvw_metres[0],
                    OSKAR_TAG_GROUP_VIS_BLOCK,
                    OSKAR_VIS_BLOCK_TAG_BASELINE_UU, block_index, status);
            oskar_binary_read_mem(h, vis->baseline_uvw_metres[1],
                    OSKAR_TAG_GROUP_VIS_BLOCK,
                    OSKAR_VIS_BLOCK_TAG_BASELINE_VV, block_index, status);
            oskar_binary_read_mem(h, vis->baseline_uvw_metres[2],
                    OSKAR_TAG_GROUP_VIS_BLOCK,
                    OSKAR_VIS_BLOCK_TAG_BASELINE_WW, block_index, status);
        }
    }
}

#ifdef __cplusplus
}
#endif
