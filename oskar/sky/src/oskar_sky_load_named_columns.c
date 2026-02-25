/*
 * Copyright (c) 2025-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "math/oskar_cmath.h"
#include "sky/oskar_sky.h"
#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_angle.h"
#include "utility/oskar_string_trim.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEG2RAD (M_PI / 180.0)
#define ARCSEC2RAD (DEG2RAD / 3600.0)
#define MAX_TOKENS 64


/*
 * Find the header format string. May take a variety of forms:
 *
 * Format = ...
 * format=...
 * # Format = ...
 * # (...) =format
 *
 * etc.
 *
 * Returns a string containing just the column names, without
 * the "# format=()" wrapper, if the header is found.
 * Returns NULL if the line is not a valid format specifier.
 */
static char* find_clean_header(char* line, int* status)
{
    int i = 0;
    char* fmt_pos = 0;
    char* p = 0;
    char* trimmed = 0;
    if (!line) return 0;

    /* Remove whitespace from both ends. */
    trimmed = oskar_string_trim(line, 0);

    /* Skip over any comment character and re-trim. */
    if (*trimmed == '#')
    {
        trimmed++;
        trimmed = oskar_string_trim(trimmed, 0);
    }

    /* Convert to lower case. */
    for (p = trimmed; *p; ++p)
    {
        *p = tolower(*p);
    }

    /* Check for the presence of "format" in the line. */
    fmt_pos = strstr(trimmed, "format");

    /* If it's not there at all, the header is invalid and there's nothing more
     * to check. Assume it's an old format file? */
    if (!fmt_pos) return 0;

    /* Check that "format" appears either at the start or the end
     * of the trimmed line. */
    const int fmt_len = 6;

    /* Check whether it's at the start. */
    const int at_start = (fmt_pos == trimmed);

    /* If it's at the end, then we require a null terminator after "format". */
    const char* after_fmt = fmt_pos + fmt_len;
    const int at_end = (*after_fmt == '\0');

    /* If it's neither at the start nor the end, the header is invalid. */
    if (!at_start && !at_end)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return 0;
    }

    /* Check for and remove any '=' character before or after "format". */
    if (at_start)
    {
        char* after = fmt_pos + fmt_len;
        while (*after && isspace(*after))
        {
            after++;
        }
        if (*after == '=')
        {
            /* Blank the '=' character. */
            *after = ' ';
        }
        else
        {
            /* There's no '=' character, so header is invalid. */
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            return 0;
        }
    }
    else /* at_end */
    {
        char* before = fmt_pos - 1;
        while (before >= trimmed && isspace(*before))
        {
            before--;
        }
        if (before >= trimmed && *before == '=')
        {
            /* Blank the '=' character. */
            *before = ' ';
        }
        else
        {
            /* There's no '=' character, so header is invalid. */
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            return 0;
        }
    }

    /* Blank out the "format" word from the header. */
    for (i = 0; i < fmt_len && fmt_pos[i]; ++i)
    {
        fmt_pos[i] = ' ';
    }

    /* Remove the new spaces at the start or end. */
    trimmed = oskar_string_trim(trimmed, 0);

    /* Blank out any brackets around what's left (keeping internal ones). */
    const size_t n = strlen(trimmed);
    if (n > 0)
    {
        if (*trimmed == '[' || *trimmed == '(') *trimmed = ' ';
        if (n > 1)
        {
            char* end = trimmed + n - 1;
            if (*end == ']' || *end == ')') *end = ' ';
        }
    }

    /* Trim the line again to remove the spaces that were used for blanking. */
    return oskar_string_trim(trimmed, 0);
}


/* Find out which columns we have from the cleaned-up header string. */
static int parse_header_columns(
        char* header_line,
        oskar_SkyColumn** column_types,
        char*** column_defaults,
        const int* status
)
{
    int capacity = 0, count = 0;
    char* p = header_line;
    char* token_start = p;
    int in_token = 0;
    int in_quotes = 0;
    char quote_char = 0;
    if (*status || !header_line || !*header_line) return 0;
    for (;;)
    {
        /* Current character. */
        const char c = *p;

        /* If we're inside a quoted default, ignore all delimiters until
         * the closing quote. */
        if (in_quotes)
        {
            if (c == quote_char) in_quotes = 0; /* End of quoted default. */
            if (c == '\0') break;

            /* Next character. */
            p++;
            continue;
        }

        /* Enter quoted-default mode if current char is a quote and the
         * previous char is '='. */
        if ((c == '"' || c == '\'') && *(p - 1) == '=')
        {
            in_quotes = 1;
            quote_char = c;

            /* Next character. */
            p++;
            continue;
        }

        /* Token boundary unless inside quotes. */
        if (c == ',' || isspace(c) || c == '\0')
        {
            /* Resize arrays if needed before appending a new token. */
            if (count >= capacity)
            {
                capacity = (capacity == 0) ? 8 : (capacity * 2);
                *column_types = (oskar_SkyColumn*) realloc(
                        *column_types, capacity * sizeof(oskar_SkyColumn)
                );
                *column_defaults = (char**) realloc(
                        *column_defaults, capacity * sizeof(char*)
                );
            }
            if (in_token)
            {
                char* name = 0;
                char* val = 0;
                *p = '\0';
                in_token = 0;
                name = oskar_string_trim(token_start, 0);

                /* Handle optional default value, after an '='. */
                val = strchr(name, '=');
                if (val)
                {
                    *val = '\0';

                    /* Also strip surrounding quotes if present. */
                    val = oskar_string_trim(val + 1, 1);
                }
                (*column_types)[count] = oskar_sky_column_type_from_name(name);
                (*column_defaults)[count] = val;
                count++;
            }
            else if (c == ',')
            {
                /* Empty column. */
                (*column_types)[count] = OSKAR_SKY_CUSTOM;
                (*column_defaults)[count] = 0;
                count++;
            }
            if (c == '\0') break;

            /* Next character. */
            p++;
            token_start = p;
            continue;
        } /* (End if token boundary.) */

        /* Start of new token (outside quotes). */
        if (!in_token)
        {
            in_token = 1;
            token_start = p;
        }
        p++;
    }
    return count;
}


/* Splits up a vector of values enclosed in brackets. */
static int parse_array_values(const char* str, double** out, int* num_out)
{
    int num = 0;

    /* No values: assume 0. */
    if (!str || !*str)
    {
        /* Don't need to check size, as num_out is always at least 4 here. */
        (*out)[0] = 0;
        return 1;
    }

    /* Single non-bracketed value. */
    if (str[0] != '[' && str[0] != '(')
    {
        /* Don't need to check size, as num_out is always at least 4 here. */
        (*out)[0] = strtod(str, 0);
        return 1;
    }

    /* Skip opening bracket. */
    str++;

    /* Find end of bracketed region. */
    const char* end_bracket = str;
    while (*end_bracket && *end_bracket != ']' && *end_bracket != ')')
    {
        end_bracket++;
    }
    const char* p = str;
    while (p < end_bracket)
    {
        /* Skip whitespace and commas. */
        while (p < end_bracket && (isspace(*p) || *p == ','))
        {
            p++;
        }
        if (p >= end_bracket) break;

        /* Parse number using strtod. */
        char* next = 0;
        const double val = strtod(p, &next);
        if (next == p)
        {
            /* Not a number: skip one char to avoid infinite loop. */
            p++;
            continue;
        }

        /* Ensure capacity. */
        if (num >= *num_out)
        {
            int new_num_out = (*num_out == 0) ? 4 : (*num_out * 2);
            double* tmp = (double*) realloc(*out, new_num_out * sizeof(double));
            *out = tmp;
            *num_out = new_num_out;
        }

        (*out)[num++] = val;
        p = next;
    }
    return num;
}


/* Row tokeniser. Split on spaces and commas, respect quotes and brackets. */
static int split_tokens(char* line, char* tokens[], int max_tokens)
{
    int count = 0;
    int bracket_depth = 0;
    int in_quotes = 0;
    char quote_char = ' ';
    char* p = line;
    char* token_start = p;
    if (count < max_tokens) tokens[count++] = token_start;
    while (*p && count < max_tokens)
    {
        /* Handle brackets and delimiters when not in quotes. */
        if (!in_quotes)
        {
            if (*p == '[' || *p == '(')
            {
                bracket_depth++;
            }
            else if ((*p == ']' || *p == ')') && bracket_depth > 0)
            {
                bracket_depth--;
            }
            else if ((*p == ',' || isspace(*p)) && bracket_depth == 0)
            {
                /* Token boundary. Add a null-terminator. */
                *p = '\0';
                p++;

                /* Skip any additional spaces. */
                while (*p && isspace(*p))
                {
                    p++;
                }

                /* Start of next token. */
                tokens[count++] = p;

                /* Already at the next character - don't increment p again. */
                continue;
            }
        }

        /* Handle quotes. Quotes override everything, including brackets. */
        if (*p == '"' || *p == '\'')
        {
            if (!in_quotes)
            {
                in_quotes = 1;
                quote_char = *p;
            }
            else if (in_quotes && *p == quote_char)
            {
                in_quotes = 0;
                quote_char = ' ';
            }
        }

        /* Check next character. */
        p++;
    }
    return count;
}


/* Save the value (or the default) in the sky model. */
static inline void set_value(
        oskar_Sky* sky,
        oskar_Mem* column,
        oskar_SkyColumn column_type,
        const char* value,
        const char* default_value,
        int row,
        int* values_size,
        double** values,
        int* status
)
{
    /* Use default if there's no value. */
    if (!value) value = default_value;

    /* Column type here is that which was detected in the file. */
    switch (column_type)
    {
    case OSKAR_SKY_CUSTOM:
        /* Ignore unknown columns. This includes name and type strings. */
        return;
    case OSKAR_SKY_RA_RAD:
        (*values)[0] = oskar_string_hours_to_radians(value, 'r', status);
        break;
    case OSKAR_SKY_DEC_RAD:
        (*values)[0] = oskar_string_degrees_to_radians(value, 'r', status);
        break;
    case OSKAR_SKY_RA_DEG:
        (*values)[0] = oskar_string_hours_to_radians(value, 'd', status);
        break;
    case OSKAR_SKY_DEC_DEG:
        (*values)[0] = oskar_string_degrees_to_radians(value, 'd', status);
        break;
    case OSKAR_SKY_MAJOR_RAD:
    case OSKAR_SKY_MINOR_RAD:
        (*values)[0] = value ? ARCSEC2RAD * strtod(value, 0) : 0.0;
        break;
    case OSKAR_SKY_PA_RAD:
    case OSKAR_SKY_POLA_RAD:
        (*values)[0] = value ? DEG2RAD * strtod(value, 0) : 0.0;
        break;
    case OSKAR_SKY_LIN_SI: /* Linear not log. */
    {
        /* Default value for LogarithmicSI is true, even if absent. */
        const char c = value ? tolower(*value) : 't';
        const double log_si = (c == 'f' || c == 'n' || c == '0') ? 0. : 1.;
        (*values)[0] = log_si > 0. ? 0. : 1.; /* Inverted for sky model. */
        break;
    }
    default:
    {
        /* All other known columns. Assume that they can be arrays. */
        int num_items = 0;
        num_items = parse_array_values(value, values, values_size);
        if (num_items == 0)
        {
            num_items = parse_array_values(default_value, values, values_size);
        }
        if (num_items == 0) return;
        if (num_items > 1)
        {
            int b = 0;
            for (; b < num_items; ++b)
            {
                if ((*values)[b] != 0.0)
                {
                    oskar_sky_set_data(
                            sky, column_type, b, row, (*values)[b], status
                    );
                }
            }
            return;
        }
        break;
    }
    }
    oskar_mem_set_element_real(column, row, (*values)[0], status);
}


oskar_Sky* oskar_sky_load_named_columns(
        const char* filename,
        int type,
        int* status
)
{
    int c = 0, n = 0;
    int array_workspace_size = 4;
    int have_ra = 0;
    int have_dec = 0;
    int have_flux = 0;
    char* buf = 0;
    char* buf_hdr = 0;
    oskar_SkyColumn* column_types = 0;
    char** column_defaults = 0;
    double* array_workspace = 0;
    char* hdr = 0;
    size_t buf_size = 0;
    size_t hdr_buf_size = 0;
    FILE* file = 0;
    oskar_Sky* sky = 0;
    oskar_Mem** cached_columns = 0;
    if (*status) return 0;

    /* Check the data type. */
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;                /* LCOV_EXCL_LINE */
        return 0;                                         /* LCOV_EXCL_LINE */
    }

    /* Open the file. */
    file = fopen(filename, "r");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;                      /* LCOV_EXCL_LINE */
        return 0;                                         /* LCOV_EXCL_LINE */
    }

    /* Find Format= header and parse it. */
    while (oskar_getline(&buf_hdr, &hdr_buf_size, file) != OSKAR_ERR_EOF)
    {
        if (*status) break;
        if ((hdr = find_clean_header(buf_hdr, status))) break;
        if (ftell(file) > 0x1000L) break; /* Only search within first 4 kiB. */
    }
    const int num_columns = parse_header_columns(
            hdr, &column_types, &column_defaults, status
    );
#if 0
    /* Debug printing of detected columns and defaults. */
    for (c = 0; c < num_columns; ++c)
    {
        printf("Column %s has default %s\n",
                oskar_sky_column_type_to_name(column_types[c]),
                column_defaults[c]
        );
    }
#endif

    /* Check for malformed header: "format" present but incorrect. */
    if (*status == OSKAR_ERR_INVALID_ARGUMENT)
    {
        oskar_log_error(
                0, "Error opening '%s': format string present "
                "but not correct.", filename
        );
    }

    /* Check if header is missing. */
    if (num_columns <= 0 && !(*status))
    {
        /* Do not print a failure message here -
         * try to load the file using the old format instead. */
        *status = OSKAR_ERR_FILE_IO;
    }

    /* Check that the required columns are present. */
    for (c = 0; c < num_columns; ++c)
    {
        oskar_SkyColumn col = column_types[c];
        if (col == OSKAR_SKY_RA_DEG || col == OSKAR_SKY_RA_RAD) have_ra++;
        if (col == OSKAR_SKY_DEC_DEG || col == OSKAR_SKY_DEC_RAD) have_dec++;
        if (col == OSKAR_SKY_I_JY) have_flux++;
    }
    if (!(*status))
    {
        if (have_ra != 1 || have_dec != 1)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(
                    0, "Error opening '%s': need one RA and one Dec column.",
                    filename
            );
        }
        if (have_flux == 0)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(
                    0, "Error opening '%s': need a Stokes I column.",
                    filename
            );
        }
    }

    /* Create an empty sky model and cache the column handles if possible. */
    sky = oskar_sky_create(type, OSKAR_CPU, 0, status);
    cached_columns = (oskar_Mem**) calloc(num_columns, sizeof(oskar_Mem*));
    array_workspace = (double*) calloc(array_workspace_size, sizeof(double));
    for (c = 0; c < num_columns; ++c)
    {
        oskar_SkyColumn column_type = column_types[c];
        /* If column is default degrees, make sure the one we store the
         * values in is in radians (convert as needed before storing). */
        if (column_type == OSKAR_SKY_RA_DEG) column_type = OSKAR_SKY_RA_RAD;
        if (column_type == OSKAR_SKY_DEC_DEG) column_type = OSKAR_SKY_DEC_RAD;
        if (column_type != OSKAR_SKY_CUSTOM)
        {
            cached_columns[c] = oskar_sky_column(sky, column_type, 0, status);
        }
    }

    /* Loop over lines in file. */
    while (oskar_getline(&buf, &buf_size, file) != OSKAR_ERR_EOF && !(*status))
    {
        char* trimmed = 0;
        char* tokens[MAX_TOKENS];

        /* Skip empty or comment lines. */
        trimmed = oskar_string_trim(buf, 0);
        if (*trimmed == '#' || *trimmed == '\0') continue;

        /* Split up the line, keeping any bracketed values together. */
        const int num_tokens = split_tokens(trimmed, tokens, MAX_TOKENS);

        /* Ensure enough space in sky model. */
        if (oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES) <= n)
        {
            const int new_size = ((2 * n) < 100) ? 100 : (2 * n);
            oskar_sky_resize(sky, new_size, status);
            if (*status) break;
        }

        /* Loop over columns in the header. */
        for (c = 0; c < num_columns; c++)
        {
            char* val = c < num_tokens ? oskar_string_trim(tokens[c], 1) : 0;
            if (val && *val == '\0') val = 0;
            set_value(
                    sky, cached_columns[c], column_types[c],
                    val, column_defaults[c], n,
                    &array_workspace_size, &array_workspace, status
            );
        }

        /* Increment source counter. */
        n++;
    }

    /* Set the size to be the actual number of elements loaded. */
    oskar_sky_resize(sky, n, status);

    /* Clean up. */
    free(array_workspace);
    free(buf);
    free(buf_hdr);
    free(cached_columns);
    free(column_defaults);
    free(column_types);
    (void) fclose(file);

    /* Check if an error occurred. */
    if (*status)
    {
        oskar_sky_free(sky, status);
        sky = 0;
    }

    /* Return a handle to the sky model. */
    return sky;
}

#ifdef __cplusplus
}
#endif
