/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_CREATE_COLUMNS_H_
#define OSKAR_SKY_CREATE_COLUMNS_H_

/**
 * @file oskar_sky_create_columns.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Creates columns in an empty sky model, based on those present in another.
 *
 * @details
 * This function creates columns in an empty sky model to match those in
 * another one. Scratch columns are also created.
 *
 * @param[in] sky          Sky model to update.
 * @param[in] src          Existing sky model containing columns.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_sky_create_columns(
        oskar_Sky* sky,
        const oskar_Sky* src,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
