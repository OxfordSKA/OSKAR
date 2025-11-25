/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_STRING_TO_ANGLE_H_
#define OSKAR_STRING_TO_ANGLE_H_

/**
 * @file oskar_string_to_angle.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Converts a string representing degrees (e.g. Dec) to radians.
 *
 * @details
 * Converts a string representing degrees (e.g. Dec) to radians.
 *
 * This handles casacore-compatible strings, e.g.:
 * - 12.34.56.789
 * - 12d34m56.789
 * - 123.456deg
 * - 123.456rad
 *
 * Values without a suffix are interpreted according to
 * the value of \p default_unit.
 *
 * @param[in,out] str The input string to parse.
 * @param[in] default_unit How to interpret default values: 'r'=rad, 'd'=deg.
 * @param[out] status Status return code.
 *
 * @return The parsed value in radians.
 */
OSKAR_EXPORT
double oskar_string_degrees_to_radians(
        const char* str,
        char default_unit,
        int* status
);

/**
 * @brief Converts a string representing hours (e.g. RA) to radians.
 *
 * @details
 * Converts a string representing hours (e.g. RA) to radians.
 *
 * This handles casacore-compatible strings, e.g.:
 * - 12:34:56.789
 * - 12h34m56.789
 * - 123.456deg
 * - 123.456rad
 *
 * Values without a suffix are interpreted according to
 * the value of \p default_unit.
 *
 * @param[in,out] str The input string to parse.
 * @param[in] default_unit How to interpret default values: 'r'=rad, 'd'=deg.
 * @param[out] status Status return code.
 *
 * @return The parsed value in radians.
 */
OSKAR_EXPORT
double oskar_string_hours_to_radians(
        const char* str,
        char default_unit,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
