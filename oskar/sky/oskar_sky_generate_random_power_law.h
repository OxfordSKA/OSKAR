/*
 * Copyright (c) 2016-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_GENERATE_RANDOM_POWER_LAW_H_
#define OSKAR_SKY_GENERATE_RANDOM_POWER_LAW_H_

/**
 * @file oskar_sky_generate_random_power_law.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Generates a uniformly distributed population with a power-law in flux.
 *
 * @details
 * Generates a uniformly distributed population with a power-law in flux.
 *
 * @param[in] precision     Precision of the sky model to create.
 * @param[in] num_sources   Number of sources over the sphere.
 * @param[in] flux_min_jy   Minimum flux, in Jy.
 * @param[in] flux_max_jy   Maximum flux, in Jy.
 * @param[in] power         Power law exponent.
 * @param[in] seed          Random generator seed.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
oskar_Sky* oskar_sky_generate_random_power_law(
        int precision,
        int num_sources,
        double flux_min_jy,
        double flux_max_jy,
        double power,
        int seed,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
