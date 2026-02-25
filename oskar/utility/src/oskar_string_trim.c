/*
 * Copyright (c) 2025-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <ctype.h>
#include <string.h>

#include "utility/oskar_string_trim.h"

#ifdef __cplusplus
extern "C" {
#endif


char* oskar_string_trim(char* str, int trim_quotes)
{
    if (!str) return str;
    while (*str && isspace(*str)) str++;
    if (!*str) return str;
    char* end = str + strlen(str) - 1;
    while (end > str && isspace(*end)) *end-- = '\0';
    if (trim_quotes && (*str == '"' || *str == '\''))
    {
        const char quote = *str++;
        end = strchr(str, quote);
        if (end) *end = '\0';
    }
    return str;
}

#ifdef __cplusplus
}
#endif
