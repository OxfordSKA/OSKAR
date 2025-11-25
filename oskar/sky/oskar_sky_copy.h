/*
 * Copyright (c) 2015-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_COPY_H_
#define OSKAR_SKY_COPY_H_

/**
 * @file oskar_sky_copy.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Make a copy of a sky model.
 *
 * @details
 * Makes a copy of the supplied sky model.
 *
 * @note
 * Note the destination model must be already allocated and large enough to
 * hold the sky model being copied into it.
 *
 * @param[out] dst         Sky model to copy into.
 * @param[in]  src         Sky model to copy from.
 * @param[in,out] status   Status return code.
*/
OSKAR_EXPORT
void oskar_sky_copy(oskar_Sky* dst, const oskar_Sky* src, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
