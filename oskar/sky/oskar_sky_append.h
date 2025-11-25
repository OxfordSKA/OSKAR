/*
 * Copyright (c) 2012-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_APPEND_H_
#define OSKAR_SKY_APPEND_H_

/**
 * @file oskar_sky_append.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Appends (copies) sources from one sky model to another.
 *
 * @details
 * This function appends source data in one sky model to those in another
 * by resizing the existing arrays and copying the data across.
 *
 * @param[out] dst Pointer to destination sky model.
 * @param[in]  src Pointer to source sky model.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
void oskar_sky_append(oskar_Sky* dst, const oskar_Sky* src, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
