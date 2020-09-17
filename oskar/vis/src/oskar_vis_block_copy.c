/*
 * Copyright (c) 2015-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "vis/private_vis_block.h"
#include "vis/oskar_vis_block.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_vis_block_copy(oskar_VisBlock* dst, const oskar_VisBlock* src,
        int* status)
{
    int i;
    if (*status) return;

    /* Copy the meta-data. */
    for (i = 0; i < 6; ++i)
        dst->dim_start_size[i] = src->dim_start_size[i];
    dst->has_auto_correlations = src->has_auto_correlations;
    dst->has_cross_correlations = src->has_cross_correlations;

    /* Copy the memory. */
    for (i = 0; i < 3; ++i)
    {
        if (oskar_mem_length(src->baseline_uvw_metres[i]) > 0)
            oskar_mem_copy(dst->baseline_uvw_metres[i],
                    src->baseline_uvw_metres[i], status);
        if (oskar_mem_length(src->station_uvw_metres[i]) > 0)
            oskar_mem_copy(dst->station_uvw_metres[i],
                    src->station_uvw_metres[i], status);
    }
    oskar_mem_copy(dst->auto_correlations, src->auto_correlations, status);
    oskar_mem_copy(dst->cross_correlations, src->cross_correlations, status);
}

#ifdef __cplusplus
}
#endif
