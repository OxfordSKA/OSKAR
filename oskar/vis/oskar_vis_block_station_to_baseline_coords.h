/*
 * Copyright (c) 2020-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_VIS_BLOCK_STATION_TO_BASELINE_COORDS_H_
#define OSKAR_VIS_BLOCK_STATION_TO_BASELINE_COORDS_H_

/**
 * @file oskar_vis_block_station_to_baseline_coords.h
 */

#include <oskar_global.h>
#include <vis/oskar_vis_header.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Calculates baseline coordinates from station coordinates.
 *
 * @details
 * Calculates baseline coordinates from station coordinates
 * in the visibility block.
 *
 * @param[in,out] vis          The visibility block.
 * @param[in] hdr              The visibility header.
 * @param[in,out] status       Status return code.
 */
OSKAR_EXPORT
void oskar_vis_block_station_to_baseline_coords(
        oskar_VisBlock* vis,
        const oskar_VisHeader* hdr,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
