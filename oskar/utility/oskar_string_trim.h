/*
 * Copyright (c) 2025-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_STRING_TRIM_H_
#define OSKAR_STRING_TRIM_H_

/**
 * @file oskar_string_trim.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Trims a C-style string of leading and trailing whitespace.
 *
 * @details
 * The input string is re-terminated in-place by placing a NULL at the first
 * whitespace character after the end, and the return value is set to point
 * to the first non-whitespace character in the string.
 *
 * If \p trim_quotes is true, then any surrounding quotes that remain after
 * trimming whitespace are also removed.
 *
 * @param[in,out] str The string to trim.
 * @param[in] trim_quotes If true, then any surrounding quotes are also trimmed.
 *
 * @return A pointer to the first non-whitespace character.
 */
OSKAR_EXPORT
char* oskar_string_trim(char* str, int trim_quotes);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
