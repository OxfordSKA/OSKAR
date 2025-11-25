/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_COLUMN_TYPE_TO_NAME_H_
#define OSKAR_SKY_COLUMN_TYPE_TO_NAME_H_

/**
 * @file oskar_sky_column_type_to_name.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns a string representation of the enumerated column type.
 *
 * @details
 * Returns a string representation of the enumerated column type.
 * This is written to the header of sky model text files.
 *
 * @param[in] column_type  Enumerated column type.
 *
 * @return String describing column type, for use in text file headers.
 */
OSKAR_EXPORT
const char* oskar_sky_column_type_to_name(oskar_SkyColumn column_type);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
