/*
 * Copyright (c) 2013-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_CREATE_H_
#define OSKAR_SKY_CREATE_H_

/**
 * @file oskar_sky_create.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Creates a new sky model.
 *
 * @details
 * This function creates a new sky model and returns a handle to it.
 *
 * The sky model must be deallocated using oskar_sky_free() when it is
 * no longer required.
 *
 * @param[in]  type         Enumerated data type of arrays.
 * @param[in]  location     Memory location of sky model.
 * @param[in]  num_sources  Number of sources initially held in the sky model.
 * @param[in,out]  status   Status return code.
 *
 * @return A handle to the new data structure.
 */
OSKAR_EXPORT
oskar_Sky* oskar_sky_create(
        int type,
        int location,
        int num_sources,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
