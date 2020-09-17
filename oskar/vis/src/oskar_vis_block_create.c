/*
 * Copyright (c) 2015-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "vis/private_vis_block.h"
#include "vis/oskar_vis_block.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_VisBlock* oskar_vis_block_create(int location, int amp_type,
        int num_times, int num_channels, int num_stations,
        int create_crosscorr, int create_autocorr, int* status)
{
    oskar_VisBlock* vis = 0;
    int i, type;

    /* Check type. */
    if (oskar_type_is_double(amp_type))
        type = OSKAR_DOUBLE;
    else if (oskar_type_is_single(amp_type))
        type = OSKAR_SINGLE;
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }
    if (!oskar_type_is_complex(amp_type))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }

    /* Allocate the structure. */
    vis = (oskar_VisBlock*) calloc(1, sizeof(oskar_VisBlock));
    if (!vis)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return 0;
    }

    /* Store metadata. */
    vis->has_auto_correlations = create_autocorr;
    vis->has_cross_correlations = create_crosscorr;

    /* Create arrays. */
    for (i = 0; i < 3; ++i)
    {
        vis->baseline_uvw_metres[i] = oskar_mem_create(
                type, location, 0, status);
        vis->station_uvw_metres[i] = oskar_mem_create(
                type, location, 0, status);
    }
    vis->auto_correlations  = oskar_mem_create(amp_type, location, 0, status);
    vis->cross_correlations = oskar_mem_create(amp_type, location, 0, status);

    /* Set dimensions. */
    oskar_vis_block_resize(vis, num_times, num_channels, num_stations, status);

    /* Return handle to structure. */
    return vis;
}

#ifdef __cplusplus
}
#endif
