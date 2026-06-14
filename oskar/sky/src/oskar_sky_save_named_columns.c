/*
 * Copyright (c) 2025-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdarg.h>
#include <stdio.h>

#include "sky/oskar_sky.h"
#include "math/oskar_cmath.h"

#define RAD2DEG 180.0 / M_PI
#define RAD2ARCSEC RAD2DEG * 3600.0

#ifdef __cplusplus
extern "C" {
#endif


static inline oskar_SkyColumn get_native_column(oskar_SkyColumn column_type)
{
    /* Translate column type in file to column type stored in sky model. */
    switch (column_type)
    {
    case OSKAR_SKY_RA_DEG:
        return OSKAR_SKY_RA_RAD;
    case OSKAR_SKY_DEC_DEG:
        return OSKAR_SKY_DEC_RAD;
    case OSKAR_SKY_SEMI_MAJOR:
        return OSKAR_SKY_MAJOR_RAD;
    case OSKAR_SKY_SEMI_MINOR:
        return OSKAR_SKY_MINOR_RAD;
    default:
        return column_type;
    }
    return column_type;                                   /* LCOV_EXCL_LINE */
}


static inline void print_file(FILE* file, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    (void) vfprintf(file, fmt, args);
    va_end(args);
}


static void print_value(
        const oskar_Sky* sky,
        FILE* file,
        oskar_SkyColumn column_type,
        int column_attrib,
        int row,
        int digits
)
{
    const oskar_SkyColumn native_column = get_native_column(column_type);
    const double val = oskar_sky_data(sky, native_column, column_attrib, row);
    switch (column_type)
    {
    case OSKAR_SKY_RA_RAD:
    case OSKAR_SKY_DEC_RAD:
        print_file(file, "%.*gdeg", digits, val * RAD2DEG);
        return;
    case OSKAR_SKY_RA_DEG:
    case OSKAR_SKY_DEC_DEG:
    case OSKAR_SKY_PA_RAD:
    case OSKAR_SKY_POLA_RAD:
        print_file(file, "%.*g", digits, val * RAD2DEG);
        return;
    case OSKAR_SKY_MAJOR_RAD:
    case OSKAR_SKY_MINOR_RAD:
        if (val > 0.0) print_file(file, "%.*g", digits, val * RAD2ARCSEC);
        return;
    case OSKAR_SKY_SEMI_MAJOR:
    case OSKAR_SKY_SEMI_MINOR:
        if (val > 0.0) print_file(file, "%.*g", digits, 0.5 * val * RAD2ARCSEC);
        return;
    case OSKAR_SKY_LIN_SI: /* Linear not log. */
        print_file(file, val > 0.0 ? "false" : "true"); /* Inverted. */
        return;
    default:
        print_file(file, "%.*g", digits, val);
        return;
    }
}


void oskar_sky_save_named_columns(
        const oskar_Sky* sky,
        const char* filename,
        int use_ska_convention,
        int use_degree_coord_column,
        int write_format_wrapper,
        int write_name,
        int write_quoted_vectors,
        int write_type,
        int* status
)
{
    int c = 0, d = 0, r = 0;
    FILE* file = 0;
    if (*status) return;

    /* Check the number of columns. */
    const int num_columns = oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS);
    if (num_columns < 1) return;
    const int precision = oskar_sky_int(sky, OSKAR_SKY_PRECISION);
    const int digits = (precision == OSKAR_DOUBLE) ? 16 : 5;
    if (use_ska_convention)
    {
        use_degree_coord_column = 1;
        write_quoted_vectors = 1;
        write_type = 0;
    }

    /* Open the output file. */
    file = fopen(filename, "w");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;                      /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }

    /* Count the number of each type of column. */
    /* Only need to store attribute for first instance of the column type. */
    int num_unique_columns = 0;
    oskar_SkyColumn* unique_columns = 0;
    int* unique_columns_attribute = 0;
    for (c = 0; c < num_columns; ++c)
    {
        oskar_SkyColumn column_type = oskar_sky_column_type(sky, c);
        const int column_attrib = oskar_sky_column_attribute(sky, c);
        int matched = 0;
        if (use_degree_coord_column)
        {
            if (column_type == OSKAR_SKY_RA_RAD)
            {
                column_type = OSKAR_SKY_RA_DEG;
            }
            if (column_type == OSKAR_SKY_DEC_RAD)
            {
                column_type = OSKAR_SKY_DEC_DEG;
            }
        }
        if (use_ska_convention)
        {
            if (column_type == OSKAR_SKY_MAJOR_RAD)
            {
                column_type = OSKAR_SKY_SEMI_MAJOR;
            }
            if (column_type == OSKAR_SKY_MINOR_RAD)
            {
                column_type = OSKAR_SKY_SEMI_MINOR;
            }
        }
        for (d = 0; d < num_unique_columns; ++d)
        {
            if (unique_columns[d] == column_type)
            {
                matched = 1;
                break;
            }
        }
        if (!matched)
        {
            num_unique_columns++;
            unique_columns = (oskar_SkyColumn*) realloc(
                    unique_columns, num_unique_columns * sizeof(oskar_SkyColumn)
            );
            unique_columns_attribute = (int*) realloc(
                    unique_columns_attribute, num_unique_columns * sizeof(int)
            );
            unique_columns[num_unique_columns - 1] = column_type;
            unique_columns_attribute[num_unique_columns - 1] = column_attrib;
        }
    }

    /* Print the format string to identify the columns. */
    if (write_format_wrapper) print_file(file, "#(");
    if (write_name)
    {
        if (use_ska_convention)
        {
            print_file(file, "component_id,source_id,");
        }
        else
        {
            print_file(file, "Name,");
        }
    }
    if (write_type) print_file(file, "Type,");
    for (c = 0; c < num_unique_columns; ++c)
    {
        print_file(file, "%s", oskar_sky_column_type_to_name(
                unique_columns[c], use_ska_convention
        ));
        if (c < num_unique_columns - 1) print_file(file, ",");
    }
    if (write_format_wrapper) print_file(file, ") = format\n");

    /* Print the number of sources as a comment. */
    const int num_sources = oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES);
    if (write_format_wrapper)
    {
        print_file(file, "# NUMBER_OF_COMPONENTS=%d\n", num_sources);
    }

    /* Get handle to columns to check for extended sources. */
    const oskar_Mem* maj = oskar_sky_column_const(sky, OSKAR_SKY_MAJOR_RAD, 0);
    const oskar_Mem* min = oskar_sky_column_const(sky, OSKAR_SKY_MINOR_RAD, 0);

    /* Loop over rows. */
    for (r = 0; r < num_sources; ++r)
    {
        if (write_name)
        {
            if (use_ska_convention)
            {
                print_file(file, "\"c%d\",\"s%d\",", r + 1, r + 1);
            }
            else
            {
                print_file(file, "\"s%d\",", r + 1);
            }
        }
        if (write_type)
        {
            /* Check to see if the source is point or Gaussian. */
            if (maj && min)
            {
                const double ma = oskar_mem_get_element(maj, r, status);
                const double mi = oskar_mem_get_element(min, r, status);
                print_file(
                        file, (ma > 0.0 && mi > 0.0) ? "GAUSSIAN," : "POINT,"
                );
            }
            else
            {
                print_file(file, "POINT,");
            }
        }

        /* Loop over unique columns. */
        for (c = 0; c < num_unique_columns; ++c)
        {
            int num_entries = 0;
            const oskar_SkyColumn column_type = unique_columns[c];
            const int num_columns_of_type = oskar_sky_num_valid_columns_of_type(
                    sky, get_native_column(column_type), r
            );
            num_entries = num_columns_of_type;
            if (use_ska_convention && column_type == OSKAR_SKY_SPEC_IDX)
            {
                /* It's a bit crazy that we need to pad the spectral index
                 * vector to 5 elements for SKA sky models,
                 * but apparently it was requested. */
                num_entries = num_columns_of_type < 5 ? 5 : num_entries;
            }

            /* If there are multiple columns of this type, print as a vector. */
            if (num_entries > 1)
            {
                print_file(file, write_quoted_vectors ? "\"[" : "[");
                for (d = 0; d < num_entries; ++d)
                {
                    if (d < num_columns_of_type)
                    {
                        print_value(sky, file, column_type, d, r, digits);
                    }
                    if (d < num_entries - 1) print_file(file, ",");
                }
                print_file(file, write_quoted_vectors ? "]\"" : "]");
            }
            else if (num_columns_of_type == 1)
            {
                const int column_attrib = unique_columns_attribute[c];
                print_value(sky, file, column_type, column_attrib, r, digits);
            }

            /* Next column. */
            if (c < num_unique_columns - 1) print_file(file, ",");
        }
        print_file(file, "\n");
    }

    /* Clean up. */
    free(unique_columns);
    free(unique_columns_attribute);
    (void) fclose(file);
}

#ifdef __cplusplus
}
#endif
