/*
 * Copyright (c) 2013-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_UPDATE_HORIZON_MASK_H_
#define OSKAR_UPDATE_HORIZON_MASK_H_

/**
 * @file oskar_update_horizon_mask.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Ensures source mask value is 1 if the source is visible.
 *
 * @details
 * This function sets the horizon mask to 1 if a source is
 * visible from a particular station.
 *
 * @param[in] num_sources The number of source positions.
 * @param[in] l           Source l-direction cosines relative to reference point.
 * @param[in] m           Source m-direction cosines relative to reference point.
 * @param[in] n           Source n-direction cosines relative to reference point.
 * @param[in] ha0_rad     Local apparent hour angle of reference point, in radians.
 * @param[in] dec0_rad    Local apparent declination of reference point, in radians.
 * @param[in] lat_rad     The observer's geodetic latitude, in radians.
 * @param[in,out] mask    The input and output mask vector.
 * @param[in,out] status  Status return code.
 */
OSKAR_EXPORT
void oskar_update_horizon_mask(int num_sources, const oskar_Mem* l,
        const oskar_Mem* m, const oskar_Mem* n,
        const double ha0_rad, const double dec0_rad, const double lat_rad,
        oskar_Mem* mask, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
