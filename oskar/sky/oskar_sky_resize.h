/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_RESIZE_H_
#define OSKAR_SKY_RESIZE_H_

/**
 * @file oskar_sky_resize.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Resizes arrays in a sky model structure.
 *
 * @details
 * This function reallocates memory used by arrays in a sky model structure,
 * preserving the existing contents.
 *
 * @param[in,out]  sky           Pointer to sky model structure.
 * @param[in]      num_sources   New number of sources.
 * @param[in,out]  status        Status return code.
 */
OSKAR_EXPORT
void oskar_sky_resize(oskar_Sky* sky, int num_sources, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
