/*
 * Copyright (c) 2014, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef OSKAR_SETTINGS_STRING_TO_ARRAY_H_
#define OSKAR_SETTINGS_STRING_TO_ARRAY_H_

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
 * This function splits a string into a sequence of integers. Splitting is
 * performed either using whitespace or a comma.
 *
 * <b>Note that the input string is corrupted on exit.</b>
 *
 * Any data after a hash '#' symbol on the line is treated as a comment and
 * ignored.
 *
 * @param[in,out] str The input string to split.
 * @param[in] n The maximum number of values to return (size of array \p data).
 * @param[out] data The array of values returned.
 *
 * @return The number of values matched (or number of array elements filled).
 */
size_t oskar_settings_string_to_array_i(char* str, size_t n, int* data);

/**
 * @brief Splits a string into numeric fields (single precision).
 *
 * @details
 * This function splits a string into a sequence of numbers. Splitting is
 * performed either using whitespace or a comma.
 *
 * <b>Note that the input string is corrupted on exit.</b>
 *
 * Any data after a hash '#' symbol on the line is treated as a comment and
 * ignored.
 *
 * @param[in,out] str The input string to split.
 * @param[in] n The maximum number of values to return (size of array \p data).
 * @param[out] data The array of values returned.
 *
 * @return The number of values matched (or number of array elements filled).
 */
size_t oskar_settings_string_to_array_f(char* str, size_t n, float* data);

/**
 * @brief Splits a string into numeric fields (double precision).
 *
 * @details
 * This function splits a string into a sequence of numbers. Splitting is
 * performed either using whitespace or a comma.
 *
 * <b>Note that the input string is corrupted on exit.</b>
 *
 * Any data after a hash '#' symbol on the line is treated as a comment and
 * ignored.
 *
 * @param[in,out] str The input string to split.
 * @param[in] n The maximum number of values to return (size of array \p data).
 * @param[out] data The array of values returned.
 *
 * @return The number of values matched (or number of array elements filled).
 */
size_t oskar_settings_string_to_array_d(char* str, size_t n, double* data);

/**
 * @brief Splits a string into sub-strings.
 *
 * @details
 * This function splits a string into an array of strings. Splitting is
 * performed either using whitespace or a comma.
 *
 * <b>Note that the input string is corrupted on exit.</b>
 *
 * Any data after a hash '#' symbol on the line is treated as a comment and
 * ignored.
 *
 * @param[in,out] str The input string to split.
 * @param[in] n The maximum number of values to return (size of array \p data).
 * @param[out] data The array of strings returned.
 *
 * @return The number of array elements filled.
 */
size_t oskar_settings_string_to_array_s(char* str, size_t n, char** data);

/**
 * @brief Splits a string into sub-strings, (re)allocating space as required.
 *
 * @details
 * This function splits a string into an array of strings. Splitting is
 * performed either using whitespace or a comma.
 *
 * The array to hold the strings is resized to be as large as necessary.
 * On entry, the parameter \p n contains the initial size of the \p data array.
 * On exit, the parameter \p n contains the final size of the \p data array.
 *
 * <b>Note that the input string is corrupted on exit.</b>
 *
 * Any data after a hash '#' symbol on the line is treated as a comment and
 * ignored.
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
size_t oskar_settings_string_to_array_realloc_s(char* str, size_t* n, char*** data);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SETTINGS_STRING_TO_ARRAY_H_ */
