/*
 * Copyright (c) 2011-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_STRING_TO_ARRAY_H_
#define OSKAR_STRING_TO_ARRAY_H_

/**
 * @file oskar_string_to_array.h
 */

#include <oskar_global.h>
#include <stddef.h> /* For size_t */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Splits a string into integer fields.
 *
 * @details
 * This function splits a string into an array of integers.
 * Characters after a hash '#' symbol on the line are ignored.
 *
 * @param[in,out] str The input string to split.
 * @param[in] n The maximum number of values to return (size of array \p data).
 * @param[out] data The array of values returned.
 *
 * @return The number of array elements filled.
 */
OSKAR_EXPORT
size_t oskar_string_to_array_i(const char* str, size_t n, int* data);

/**
 * @brief Splits a string into numeric fields (double precision).
 *
 * @details
 * This function splits a string into an array of doubles.
 * Characters after a hash '#' symbol on the line are ignored.
 *
 * <b>Note that the input string is corrupted on exit.</b>
 *
 * @param[in,out] str The input string to split.
 * @param[in] n The maximum number of values to return (size of array \p data).
 * @param[out] data The array of values returned.
 *
 * @return The number of array elements filled.
 */
OSKAR_EXPORT
size_t oskar_string_to_array_d(char* str, size_t n, double* data);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
