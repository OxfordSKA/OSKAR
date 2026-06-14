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


char* oskar_string_trim(char* str, int trim_quotes, int trim_brackets)
{
    if (!str) return str;
    while (*str && isspace(*str)) str++;
    if (!*str) return str;
    char* end = str + strlen(str) - 1;
    while (end > str && isspace(*end)) *end-- = '\0';
    if (trim_quotes && (*str == '"' || *str == '\'' || *str == '`'))
    {
        const char quote = *str++;
        end = strrchr(str, quote);
        if (end) *end = '\0';
        return oskar_string_trim(str, trim_quotes, trim_brackets);
    }
    if (trim_brackets && (*str == '(' || *str == '[' || *str == '{'))
    {
        char end_bracket = '\0';
        const char bracket = *str++;
        switch (bracket)
        {
        case '(':
            end_bracket =')';
            break;
        case '[':
            end_bracket =']';
            break;
        case '{':
            end_bracket ='}';
            break;
        default:                                          /* LCOV_EXCL_LINE */
            break;                                        /* LCOV_EXCL_LINE */
        }
        end = strrchr(str, end_bracket);
        if (end) *end = '\0';
        return oskar_string_trim(str, trim_quotes, trim_brackets);
    }
    return str;
}

#ifdef __cplusplus
}
#endif
