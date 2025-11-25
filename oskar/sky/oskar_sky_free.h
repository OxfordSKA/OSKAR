/*
 * Copyright (c) 2012-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_FREE_H_
#define OSKAR_SKY_FREE_H_

/**
 * @file oskar_sky_free.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Frees memory held by the sky model.
 *
 * @param[in,out]  model    Pointer to sky model.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
void oskar_sky_free(oskar_Sky* model, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
