/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_HORIZON_CLIP_H_
#define OSKAR_SKY_HORIZON_CLIP_H_

/**
 * @file oskar_sky_horizon_clip.h
 */

#include "oskar_global.h"
#include "telescope/oskar_telescope.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Compacts a sky model into another one by removing sources below the
 * horizon of all stations.
 *
 * @details
 * Copies sources into another sky model that are above the horizon of
 * stations.
 *
 * @param[out] out          The output sky model.
 * @param[in]  in           The input sky model.
 * @param[in]  telescope    The telescope model.
 * @param[in]  gast_rad     The Greenwich Apparent Sidereal Time, in radians.
 * @param[in]  work         Work arrays.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
void oskar_sky_horizon_clip(
        oskar_Sky* out,
        const oskar_Sky* in,
        const oskar_Telescope* telescope,
        double gast_rad,
        oskar_StationWork* work,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
