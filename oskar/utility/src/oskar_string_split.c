/*
 * Copyright (c) 2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utility/oskar_string_split.h"

#ifdef __cplusplus
extern "C" {
#endif


static void add_token(
        int* list_size,
        char*** list,
        int* count,
        char* token_start,
        int* status
)
{
    if (*list_size >= 100000)
    {
        /* Catch anything too crazy. */
        printf(                                           /* LCOV_EXCL_LINE */
                "List size is too large: %d\n",           /* LCOV_EXCL_LINE */
                *list_size                                /* LCOV_EXCL_LINE */
        );                                                /* LCOV_EXCL_LINE */
        *status = OSKAR_ERR_INVALID_ARGUMENT;             /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }
    if (*list_size < *count + 1)
    {
        *list_size = *count + 1;
        *list = (char**) realloc(*list, (size_t) *list_size * sizeof(char**));
        if (!*list)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;     /* LCOV_EXCL_LINE */
            return;                                       /* LCOV_EXCL_LINE */
        }
    }
    (*list)[(*count)++] = token_start;
}


static inline int is_opening_bracket(char c)
{
    return c == '(' || c == '[' || c == '{';
}


static inline int is_closing_bracket(char c)
{
    return c == ')' || c == ']' || c == '}';
}


int oskar_string_split(
        char* line,
        int* list_size,
        char*** list,
        int split_on_equals,
        int* status
)
{
    int bracket_depth = 0;
    int count = 0;
    int currently_empty = 1;
    int in_equals = 0;
    int in_token = 0;
    int separator_seen = 0;
    char quote_char = '\0';
    char* p = line;
    if (!line || !list || !list_size) *status = OSKAR_ERR_INVALID_ARGUMENT;
    for (; *status == 0; ++p)
    {
        if (quote_char)
        {
            /* Already in quotes. */
            if (*p == '\0')
            {
                *status = OSKAR_ERR_INVALID_ARGUMENT; /* Line ends too soon. */
            }
            else if (*p == quote_char)
            {
                quote_char = '\0'; /* At closing quote. */
            }
            else if (!isspace(*p) && !isprint(*p))
            {
                *status = OSKAR_ERR_INVALID_ARGUMENT; /* Non-ASCII char. */
            }
        }
        else if (bracket_depth > 0)
        {
            /* Already in brackets. */
            if (*p == '\0')
            {
                *status = OSKAR_ERR_INVALID_ARGUMENT; /* Line ends too soon. */
            }
            else if (is_opening_bracket(*p))
            {
                ++bracket_depth;
            }
            else if (is_closing_bracket(*p))
            {
                --bracket_depth;
            }
            else if (!isspace(*p) && !isprint(*p))
            {
                *status = OSKAR_ERR_INVALID_ARGUMENT; /* Non-ASCII char. */
            }
        }
        else if (*p == '\0' || *p == '#')
        {
            /* Handle end of string or start of comment. */
            /* Add empty token if required. */
            if (separator_seen && currently_empty)
            {
                add_token(list_size, list, &count, p, status);
            }
            *p = '\0';
            break; /* Stop parsing. */
        }
        else if (*p == '=')
        {
            /* Handle equals. */
            in_equals = 1;
            if (split_on_equals)
            {
                /* Terminate current token and add the final one. */
                *p = '\0';
                p++;
                while (isspace(*p))
                {
                    p++;
                }
                add_token(list_size, list, &count, p, status);
                break; /* Stop parsing. */
            }
        }
        else if (*p == ',')
        {
            /* Handle comma. Add empty token if required. */
            if (currently_empty) add_token(list_size, list, &count, p, status);

            /* Terminate token and reset state. */
            *p = '\0';
            in_equals = 0;
            in_token = 0;
            currently_empty = 1;
            separator_seen = 1;
        }
        else if (isspace(*p))
        {
            /* Handle space. */
            if (in_token && !in_equals)
            {
                if (!split_on_equals)
                {
                    /* Check to see if next non-space is '='. */
                    char* q = p;
                    while (isspace(*q))
                    {
                        q++;
                    }
                    if (*q == '=')
                    {
                        in_equals = 1;
                        p = q; /* Jump ahead. */
                    }
                }
                if (!in_equals)
                {
                    /* Terminate token at current position. */
                    *p = '\0';
                    in_token = 0;
                }
            }
        }
        else if (isprint(*p))
        {
            /* Handle any other printable character. */
            /* If not currently in a token, start one at current position. */
            if (!in_token)
            {
                in_token = 1;
                add_token(list_size, list, &count, p, status);
            }
            if (*p == '"' || *p == '\'' || *p == '`')
            {
                quote_char = *p; /* At opening quote. */
            }
            else if (is_opening_bracket(*p))
            {
                ++bracket_depth;
            }
            currently_empty = 0;
            in_equals = 0;
        }
        else
        {
            /* Set error flag if character is not printable. */
            *status = OSKAR_ERR_INVALID_ARGUMENT;
        }
    }
    return count;
}

#ifdef __cplusplus
}
#endif
