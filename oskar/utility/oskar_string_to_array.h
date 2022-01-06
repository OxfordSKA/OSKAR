/*
 * Copyright (c) 2011-2022, The OSKAR Developers.
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

/**
 * @brief Splits a string into sub-strings.
 *
 * @details
 * This function splits a string into an array of strings. Splitting is
 * performed either using whitespace or a comma.
 * Characters after a hash '#' symbol on the line are ignored.
 *
 * <b>Note that the input string is corrupted on exit.</b>
 *
 * @param[in,out] str The input string to split.
 * @param[in] n The maximum number of values to return (size of array \p data).
 * @param[out] data The array of strings returned.
 *
 * @return The number of array elements filled.
 */
OSKAR_EXPORT
size_t oskar_string_to_array_s(char* str, size_t n, char** data);

/**
 * @brief Splits a string into numeric fields (double precision).
 *
 * @details
 * This function splits a string into an array of doubles.
 * Characters after a hash '#' symbol on the line are ignored.
 *
 * The array to hold the data is resized to be as large as necessary.
 * On entry, the parameter \p n contains the initial size of the \p data array.
 * On exit, the parameter \p n contains the final size of the \p data array.
 *
 * <b>Note that the input string is corrupted on exit.</b>
 *
 * WARNING: This function allocates memory for the array,
 * which must be freed by the caller when no longer needed.
 *
 * @param[in,out] str    The input string to split.
 * @param[in,out] n      The initial/final size of array at \p data.
 * @param[in,out] data   Pointer to the array to resize.
 *
 * @return The number of array elements filled.
 */
OSKAR_EXPORT
size_t oskar_string_to_array_realloc_d(char* str, size_t* n, double** data);

/**
 * @brief Splits a string into sub-strings, (re)allocating space as required.
 *
 * @details
 * This function splits a string into an array of strings. Splitting is
 * performed either using whitespace or a comma.
 * Characters after a hash '#' symbol on the line are ignored.
 *
 * The array to hold the strings is resized to be as large as necessary.
 * On entry, the parameter \p n contains the initial size of the \p data array.
 * On exit, the parameter \p n contains the final size of the \p data array.
 *
 * <b>Note that the input string is corrupted on exit.</b>
 *
 * WARNING: This function allocates memory for the list of string pointers,
 * which must be freed by the caller when no longer needed.
 *
 * Example use:
 * @code
   char** list = 0;
   int n = 0;
   char line[] = "A text string to split.";
   oskar_string_to_array_realloc_s(line, &n, &list);
   // n == 5;
   // strcmp(list[0], "A") == 0;
   // strcmp(list[1], "text") == 0;
   // strcmp(list[2], "string") == 0;
   // strcmp(list[3], "to") == 0;
   // strcmp(list[4], "split.") == 0;
   free(list);
   @endcode
 *
 * @param[in,out] str    The input string to split.
 * @param[in,out] n      The initial/final size of array at \p data.
 * @param[in,out] data   Pointer to the string array to resize.
 *
 * @return The number of array elements filled.
 */
OSKAR_EXPORT
size_t oskar_string_to_array_realloc_s(char* str, size_t* n, char*** data);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
