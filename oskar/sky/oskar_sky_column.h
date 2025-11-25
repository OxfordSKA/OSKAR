/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_COLUMN_H_
#define OSKAR_SKY_COLUMN_H_

/**
 * @file oskar_sky_column.h
 */

#include "oskar_global.h"
#include "mem/oskar_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns a handle to a sky model data column, creating it if needed.
 *
 * @details
 * Returns a handle to a sky model data column.
 *
 * If a column with the specified type and attribute does not exist
 * in the sky model, it will be created first.
 *
 * @param[in] sky              Pointer to sky model.
 * @param[in] column_type      Enumerated column type.
 * @param[in] column_attribute Optional column attribute: Set to 0 if unused.
 * @param[in,out] status       Status return code.
 */
OSKAR_EXPORT
oskar_Mem* oskar_sky_column(
        oskar_Sky* sky,
        oskar_SkyColumn column_type,
        int column_attribute,
        int* status
);

/**
 * @brief Returns a handle to a sky model data column (const version).
 *
 * @details
 * Returns a handle to a sky model data column (const version).
 *
 * A null pointer is returned if a column of the specified type and attribute
 * does not exist in the sky model.
 *
 * @param[in] sky              Pointer to sky model.
 * @param[in] column_type      Enumerated column type.
 * @param[in] column_attribute Optional column attribute: Set to 0 if unused.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_sky_column_const(
        const oskar_Sky* sky,
        oskar_SkyColumn column_type,
        int column_attribute
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
