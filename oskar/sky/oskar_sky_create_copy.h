/*
 * Copyright (c) 2013-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_CREATE_COPY_H_
#define OSKAR_SKY_CREATE_COPY_H_

/**
 * @file oskar_sky_create_copy.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Creates a new sky model by copying an existing one.
 *
 * @details
 * This function creates a new sky model by copying an existing one to
 * the specified location, and returns a handle to the new copy.
 *
 * The sky model must be deallocated using oskar_sky_free() when it is
 * no longer required.
 *
 * @param[in]  src          Pointer to existing sky model to copy.
 * @param[in]  location     Location of new sky model.
 * @param[in,out]  status   Status return code.
 *
 * @return A handle to the new data structure.
 */
OSKAR_EXPORT
oskar_Sky* oskar_sky_create_copy(
        const oskar_Sky* src,
        int location,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
