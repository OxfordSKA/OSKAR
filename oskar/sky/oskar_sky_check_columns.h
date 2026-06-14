/*
 * Copyright (c) 2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_CHECK_COLUMNS_H_
#define OSKAR_SKY_CHECK_COLUMNS_H_

/**
 * @file oskar_sky_check_columns.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Checks for consistent source data in a sky model.
 *
 * @details
 * This function checks column data in a sky model after it has been loaded,
 * and sets the status code if an invalid parameter combination is found.
 *
 * @param[in] sky          Sky model to check.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_sky_check_columns(const oskar_Sky* sky, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
