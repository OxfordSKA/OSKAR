/*
 * Copyright (c) 2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_STRING_SPLIT_H_
#define OSKAR_STRING_SPLIT_H_

/**
 * @file oskar_string_split.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Splits a string by spaces and commas, respecting equals, quotes and brackets.
 *
 * @details
 * This function splits a string into an array of string tokens.
 * Splitting is performed in-place, using whitespace and commas, but splitting
 * is not performed inside quoted or bracketed regions in the string.
 * The equals character is also treated specially:
 * - if \p split_on_equals is false, then an equals character will keep
 *   neighbouring tokens together if separated only by spaces;
 * - if \p split_on_equals is true, then splitting will stop after the first
 *   non-quoted, non-bracketed equals character is encountered.
 * Characters after a hash '#' symbol on the line are ignored.
 *
 * The array to hold pointers to the sub-strings is resized to be
 * as large as necessary.
 * On entry, the parameter \p list_size contains the initial size of
 * the \p list array.
 * On exit, the parameter \p list_size contains the final size of
 * the \p list array.
 * The return value is the number of tokens actually found in the input string.
 *
 * <b>
 * Note that the input string is tokenised in-place, and
 * pointers to the sub-string tokens refer to locations in the input string.
 * </b>
 *
 * Note also that this function allocates memory for the list of sub-string
 * pointers, which must be freed by the caller when no longer needed.
 *
 * Example use:
 * @code
   char** list = 0;
   int list_size = 0, num_found = 0, split_on_equals = 0, status = 0;
   char line1[] = "A text string to split.";
   num_found = oskar_string_split(
           line1, &list_size, &list, split_on_equals, &status
   );
   // list_size == 5;
   // num_found == 5;
   // strcmp(list[0], "A") == 0;
   // strcmp(list[1], "text") == 0;
   // strcmp(list[2], "string") == 0;
   // strcmp(list[3], "to") == 0;
   // strcmp(list[4], "split.") == 0;

   char line2[] = "1.1, 2.2 [0.1, 0.2, 0.3] , (0.4 0.5)   [0.6,0.7]  ";
   num_found = oskar_string_split(
           line2, &list_size, &list, split_on_equals, &status
   );
   // list_size == 5;
   // num_found == 5;
   // strcmp(list[0], "1.1") == 0;
   // strcmp(list[1], "2.2") == 0;
   // strcmp(list[2], "[0.1, 0.2, 0.3]") == 0;
   // strcmp(list[3], "(0.4 0.5)") == 0;
   // strcmp(list[4], "[0.6,0.7]") == 0;
   free(list);
   @endcode
 *
 * @param[in,out] line The string to split, modified in-place.
 * @param[in,out] list_size The initial/final size of array at \p list.
 * @param[in,out] list Pointer to the string array.
 * @param[in] split_on_equals If true, stop splitting after the first equals.
 * @param[in,out] status Status return code.
 *
 * @return The number of tokens (sub-strings) found in the input string.
 */
OSKAR_EXPORT
int oskar_string_split(
        char* line,
        int* list_size,
        char*** list,
        int split_on_equals,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
