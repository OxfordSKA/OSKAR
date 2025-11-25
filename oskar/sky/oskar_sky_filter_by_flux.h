/*
 * Copyright (c) 2012-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_FILTER_BY_FLUX_H_
#define OSKAR_SKY_FILTER_BY_FLUX_H_

/**
 * @file oskar_sky_filter_by_flux.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Removes sources outside a given flux range in Stokes I.
 *
 * @details
 * This function removes sources from a sky model that lie outside a given
 * Stokes-I flux range specified by \p min_flux and \p max_flux .
 *
 * @param[in,out] sky    Pointer to sky model.
 * @param[in] min_flux   Minimum Stokes I flux.
 * @param[in] max_flux   Maximum Stokes I flux.
 * @param[in,out] status Status return code.
 */
OSKAR_EXPORT
void oskar_sky_filter_by_flux(
        oskar_Sky* sky,
        double min_flux,
        double max_flux,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
