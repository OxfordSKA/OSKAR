/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_COLUMN_TYPE_FROM_NAME_H_
#define OSKAR_SKY_COLUMN_TYPE_FROM_NAME_H_

/**
 * @file oskar_sky_column_type_from_name.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns the column type by comparing against a set of known strings.
 *
 * @details
 * Returns the column type by comparing against a set of known strings.
 *
 * @param[in] name             String name to check.
 *
 * @return Enumerated column type.
 */
OSKAR_EXPORT
oskar_SkyColumn oskar_sky_column_type_from_name(const char* name);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
