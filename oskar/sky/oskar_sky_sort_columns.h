/*
 * Copyright (c) 2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_SORT_COLUMNS_H_
#define OSKAR_SKY_SORT_COLUMNS_H_

/**
 * @file oskar_sky_sort_columns.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Sorts columns in a sky model.
 *
 * @details
 * This function sorts columns in a sky model after it has been loaded.
 * It also ensures that column attributes have been set up appropriately.
 *
 * @param[in] sky          Sky model to sort.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_sky_sort_columns(oskar_Sky* sky, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
