/*
 * Copyright (c) 2014-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_COPY_SOURCE_DATA_H_
#define OSKAR_SKY_COPY_SOURCE_DATA_H_

/**
 * @file oskar_sky_copy_source_data.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Copies sources above the horizon to a new sky model.
 *
 * @details
 * Copies sources above the horizon to a new sky model.
 *
 * @param[in] in            Input sky model.
 * @param[in] horizon_mask  Truth array. Value is 1 for a visible source.
 * @param[in] indices       Output of prefix sum on \p horizon_mask.
 * @param[in,out] out       Output sky model.
 * @param[in,out] status    Status return code.
 */
OSKAR_EXPORT
void oskar_sky_copy_source_data(
        const oskar_Sky* in,
        const oskar_Mem* horizon_mask,
        const oskar_Mem* indices,
        oskar_Sky* out,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
