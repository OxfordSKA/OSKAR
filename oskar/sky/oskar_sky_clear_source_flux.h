/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_CLEAR_SOURCE_FLUX_H_
#define OSKAR_SKY_CLEAR_SOURCE_FLUX_H_

/**
 * @file oskar_sky_clear_source_flux.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Clears all source flux values for a specific source.
 *
 * @details
 * Clears (sets to zero) all source flux values for a specific source.
 *
 * @param[in] sky              Pointer to sky model.
 * @param[in] index            Source index to clear flux values for.
 * @param[in,out] status       Status return code.
 */
OSKAR_EXPORT
void oskar_sky_clear_source_flux(oskar_Sky* sky, int index, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
