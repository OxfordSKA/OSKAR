/*
 * Copyright (c) 2025-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log/oskar_log.h"
#include "math/oskar_cmath.h"
#include "sky/oskar_sky.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_getline.h"
#include "utility/oskar_string_split.h"
#include "utility/oskar_string_to_angle.h"
#include "utility/oskar_string_trim.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEG2RAD (M_PI / 180.0)
#define RAD2DEG (180.0 / M_PI)
#define ARCSEC2RAD (DEG2RAD / 3600.0)


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
    trimmed = oskar_string_trim(line, 0, 0);

    /* Skip over any comment character and re-trim. */
    if (*trimmed == '#')
    {
        trimmed++;
        trimmed = oskar_string_trim(trimmed, 0, 0);
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

    /* Trim the line again to remove the spaces and any enclosing brackets. */
    return oskar_string_trim(trimmed, 0, 1);
}


/* Find out which columns we have from the cleaned-up header string. */
static int parse_header(
        char* header_line,
        oskar_SkyColumn** column_types,
        char*** column_defaults,
        double** column_suffixes,
        int* status
)
{
    int i = 0, j = 0, list1_size = 0, list2_size = 0;
    char **list1 = 0, **list2 = 0;
    double suffix = 0.0;
    if (*status || !header_line || !*header_line) return 0;

    /* Split up the header string into fields. */
    const int num_fields = oskar_string_split(
            header_line, &list1_size, &list1, 0, status
    );
    if (*status || num_fields <= 0)
    {
        free(list1);
        return 0;
    }

    /* Resize the output arrays. */
    *column_types = (oskar_SkyColumn*) realloc(
            *column_types, num_fields * sizeof(oskar_SkyColumn)
    );
    *column_defaults = (char**) realloc(
            *column_defaults, num_fields * sizeof(char*)
    );
    *column_suffixes = (double*) realloc(
            *column_suffixes, num_fields * sizeof(double)
    );

    /* Process each field. */
    for (i = 0; i < num_fields; ++i)
    {
        /* Split each field name on equals to check for a default. */
        const int num_parts = oskar_string_split(
                list1[i], &list2_size, &list2, 1, status
        );
        for (j = 0; j < num_parts; ++j)
        {
            list2[j] = oskar_string_trim(list2[j], 1, 0); /* Trim quotes. */
        }
        if (num_parts > 0)
        {
            (*column_types)[i] = oskar_sky_column_type_from_name(
                    list2[0], &suffix
            );
            (*column_defaults)[i] = num_parts > 1 ? list2[1] : 0;
            (*column_suffixes)[i] = suffix;
        }
        else
        {
            (*column_types)[i] = OSKAR_SKY_CUSTOM;
            (*column_defaults)[i] = 0;
            (*column_suffixes)[i] = 0.0;
        }
    }
    free(list1);
    free(list2);
    return num_fields;
}


/* Splits up a vector of values enclosed in brackets. */
static int parse_array_values(const char* str, double** out, int* num_out)
{
    int num = 0;
    if (!str || !*str) return 0;

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


/* Save the value (or the default) in the sky model. */
static inline void set_value(
        oskar_Sky* sky,
        oskar_SkyColumn column_type,
        const char* value,
        const char* default_value,
        int row,
        int* values_size,
        double** values,
        int* status
)
{
    int b = 0, num_items = 1;
    double scaling = 1.0;
    const double tol = 1e-4;
    if (*status) return;

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
        if ((*values)[0] > 2 * M_PI + tol || (*values)[0] < -2 * M_PI - tol)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(
                    0, "Source %d has a Right Ascension (%.4f radians) "
                    "which is out of range. Check the units of "
                    "the RA and Dec columns.", row, (*values)[0]
            );
        }
        break;
    case OSKAR_SKY_DEC_RAD:
        (*values)[0] = oskar_string_degrees_to_radians(value, 'r', status);
        if ((*values)[0] > M_PI / 2 + tol || (*values)[0] < -M_PI / 2 - tol)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(
                    0, "Source %d has a Declination (%.4f radians) "
                    "which is out of range. Check the units of "
                    "the RA and Dec columns.", row, (*values)[0]
            );
        }
        break;
    case OSKAR_SKY_RA_DEG:
        (*values)[0] = oskar_string_hours_to_radians(value, 'd', status);
        if ((*values)[0] > 2 * M_PI + tol || (*values)[0] < -2 * M_PI - tol)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(
                    0, "Source %d has a Right Ascension (%.4f degrees) "
                    "which is out of range.", row, (*values)[0] * RAD2DEG
            );
        }
        break;
    case OSKAR_SKY_DEC_DEG:
        (*values)[0] = oskar_string_degrees_to_radians(value, 'd', status);
        if ((*values)[0] > M_PI / 2 + tol || (*values)[0] < -M_PI / 2 - tol)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(
                    0, "Source %d has a Declination (%.4f degrees) "
                    "which is out of range.", row, (*values)[0] * RAD2DEG
            );
        }
        break;
    case OSKAR_SKY_LIN_SI: /* Linear not log. */
    {
        /* Default value for LogarithmicSI is true, even if absent. */
        if (!value || !*value)
        {
            num_items = 0;
        }
        else
        {
            const char c = tolower(*value);
            const double log_si = (c == 'f' || c == 'n' || c == '0') ? 0. : 1.;
            (*values)[0] = log_si > 0. ? 0. : 1.; /* Inverted for sky model. */
        }
        break;
    }
    default:
    {
        /* All other known columns. Assume that they can be arrays. */
        switch (column_type)
        {
        case OSKAR_SKY_MAJOR_RAD:
        case OSKAR_SKY_MINOR_RAD:
            scaling = ARCSEC2RAD;
            break;
        case OSKAR_SKY_SEMI_MAJOR:
        case OSKAR_SKY_SEMI_MINOR:
            scaling = 2 * ARCSEC2RAD;
            break;
        case OSKAR_SKY_PA_RAD:
        case OSKAR_SKY_POLA_RAD:
            scaling = DEG2RAD;
            break;
        default:
            break;
        }
        num_items = parse_array_values(value, values, values_size);
        if (num_items == 0)
        {
            num_items = parse_array_values(default_value, values, values_size);
        }
        break;
    }
    }
    if (column_type == OSKAR_SKY_RA_DEG) column_type = OSKAR_SKY_RA_RAD;
    if (column_type == OSKAR_SKY_DEC_DEG) column_type = OSKAR_SKY_DEC_RAD;
    if (column_type == OSKAR_SKY_SEMI_MAJOR) column_type = OSKAR_SKY_MAJOR_RAD;
    if (column_type == OSKAR_SKY_SEMI_MINOR) column_type = OSKAR_SKY_MINOR_RAD;
    const int column_count = oskar_sky_num_valid_columns_of_type(
            sky, column_type, row
    );
    for (b = 0; b < num_items; ++b)
    {
        oskar_sky_set_data(
                sky, column_type, b + column_count, row,
                (*values)[b] * scaling, status
        );
    }
}


static void check_create_ref_freq_columns(
        oskar_Sky* sky,
        int num_columns,
        oskar_SkyColumn* column_types,
        const double* column_suffixes,
        int* status
)
{
    int c = 0, i = 0;
    int num_ref_freq = 0, num_stokes_i = 0;
    if (*status) return;
    num_ref_freq = oskar_sky_num_columns_of_type(sky, OSKAR_SKY_REF_HZ);
    num_stokes_i = oskar_sky_num_columns_of_type(sky, OSKAR_SKY_I_JY);
    const int capacity = oskar_sky_int(sky, OSKAR_SKY_CAPACITY);
    if (num_ref_freq == 0 && num_stokes_i >= 1)
    {
        /* Create the reference frequency columns using Stokes-I suffixes. */
        for (c = 0; c < num_columns; ++c)
        {
            if (column_types[c] == OSKAR_SKY_I_JY && column_suffixes[c] > 0.0)
            {
                for (i = 0; i < capacity; ++i)
                {
                    oskar_sky_set_data(
                            sky, OSKAR_SKY_REF_HZ, num_ref_freq, i,
                            column_suffixes[c] * 1e6, status
                    );
                }
                num_ref_freq++;
            }
        }
    }
}


oskar_Sky* oskar_sky_load_named_columns(
        const char* filename,
        int type,
        int* status
)
{
    int c = 0, n = 0;
    int array_workspace_size = 4;
    int line_counter = 0;
    char* buf = 0;
    char* buf_hdr = 0;
    oskar_SkyColumn* column_types = 0;
    char** column_defaults = 0;
    double* column_suffixes = 0;
    double* array_workspace = 0;
    char* hdr = 0;
    char** tokens = 0;
    int tokens_size = 0;
    size_t buf_size = 0;
    size_t hdr_buf_size = 0;
    FILE* file = 0;
    oskar_Sky* sky = 0;
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
        line_counter++;
        if (*status) break;
        if ((hdr = find_clean_header(buf_hdr, status))) break;
        if (ftell(file) > 0x1000L) break; /* Only search within first 4 kiB. */
    }
    const int num_columns = parse_header(
            hdr, &column_types, &column_defaults, &column_suffixes, status
    );
#if 0
    /* Debug printing of detected columns and defaults. */
    for (c = 0; c < num_columns; ++c)
    {
        printf("Column %s has default %s\n",
                oskar_sky_column_type_to_name(column_types[c], 0),
                column_defaults[c]
        );
    }
#endif

    /* Check for malformed header: "format" present but incorrect. */
    if (*status == OSKAR_ERR_INVALID_ARGUMENT)
    {
        oskar_log_error(
                0, "Error opening sky model file '%s': "
                "Format string present but not correct.", filename
        );
    }

    /* Check if header is missing. */
    if (num_columns <= 0 && !(*status))
    {
        /* Do not print a failure message here -
         * try to load the file using the old format instead. */
        *status = OSKAR_ERR_FILE_IO;
        oskar_log_warning(
                0, "Unknown sky model file header: "
                "Assuming old fixed-format sky model file."
        );
    }

    /* Create an empty sky model. */
    sky = oskar_sky_create(type, OSKAR_CPU, 0, status);
    array_workspace = (double*) calloc(array_workspace_size, sizeof(double));

    /* Loop over lines in file. */
    while (oskar_getline(&buf, &buf_size, file) != OSKAR_ERR_EOF && !(*status))
    {
        /* Skip empty or comment lines. */
        char* trimmed = oskar_string_trim(buf, 0, 0);
        line_counter++;
        if (*trimmed == '#' || *trimmed == '\0') continue;

        /* Split up the line, keeping any bracketed values together. */
        const int num_tokens = oskar_string_split(
                trimmed, &tokens_size, &tokens, 0, status
        );
        if (*status == OSKAR_ERR_INVALID_ARGUMENT)
        {
            oskar_log_error(
                    0, "Error opening sky model file '%s': "
                    "Malformed line at line %d. "
                    "Check closing quotes and closing brackets.",
                    filename, line_counter
            );
            break;
        }
        if (num_tokens > 0 && num_tokens != num_columns)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(
                    0, "Error opening sky model file '%s': "
                    "Number of fields at line %d does not match the "
                    "expected number of columns: %d found, but %d expected.",
                    filename, line_counter, num_tokens, num_columns
            );
            break;
        }

        /* Ensure enough space in sky model. */
        if (oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES) <= n)
        {
            const int new_size = ((2 * n) < 128) ? 128 : (2 * n);
            oskar_sky_resize(sky, new_size, status);
            if (*status) break;
        }

        /* Loop over columns in the header. */
        for (c = 0; c < num_columns; c++)
        {
            char* val = oskar_string_trim(tokens[c], 1, 0);
            if (val && *val == '\0') val = 0;
            set_value(
                    sky, column_types[c], val, column_defaults[c], n,
                    &array_workspace_size, &array_workspace, status
            );
        }

        /* Increment source counter. */
        n++;
    }

    /* Set the size to be the actual number of elements loaded. */
    oskar_sky_resize(sky, n, status);

    /* Add in the reference frequency columns if required/possible. */
    check_create_ref_freq_columns(
            sky, num_columns, column_types, column_suffixes, status
    );

    /* Clean up. */
    free(array_workspace);
    free(buf);
    free(buf_hdr);
    free(column_defaults);
    free(column_types);
    free(column_suffixes);
    free(tokens);
    (void) fclose(file);

    /* Sort the columns and check for consistent source parameters. */
    oskar_sky_sort_columns(sky, status);
    oskar_sky_check_columns(sky, status);

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
