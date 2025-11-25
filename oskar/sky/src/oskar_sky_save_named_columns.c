/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdio.h>

#include "sky/oskar_sky.h"
#include "math/oskar_cmath.h"

#define RAD2DEG 180.0 / M_PI
#define RAD2ARCSEC RAD2DEG * 3600.0

#ifdef __cplusplus
extern "C" {
#endif


static void print_value(
        const oskar_Sky* sky,
        FILE* file,
        oskar_SkyColumn column_type,
        int column_attribute,
        int use_degree_coord_column,
        int row
)
{
    double value = oskar_sky_data(sky, column_type, column_attribute, row);
    switch (column_type)
    {
    case OSKAR_SKY_RA_RAD:
    case OSKAR_SKY_DEC_RAD:
        (void) fprintf(file, use_degree_coord_column ? "%.15g" : "%.15gdeg",
                value * RAD2DEG
        );
        return;
    case OSKAR_SKY_MAJOR_RAD:
    case OSKAR_SKY_MINOR_RAD:
        if (value > 0.0) (void) fprintf(file, "%.15g", value * RAD2ARCSEC);
        return;
    case OSKAR_SKY_PA_RAD:
    case OSKAR_SKY_POLA_RAD:
        if (value != 0.0) (void) fprintf(file, "%.15g", value * RAD2DEG);
        return;
    case OSKAR_SKY_LIN_SI: /* Linear not log. */
        (void) fprintf(file, value > 0 ? "false" : "true"); /* Inverted. */
        return;
    default:
        (void) fprintf(file, "%.15g", value);
        return;
    }
}


void oskar_sky_save_named_columns(
        const oskar_Sky* sky,
        const char* filename,
        int use_degree_coord_column,
        int write_name,
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

    /* Open the output file. */
    file = fopen(filename, "w");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;                      /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }

    /* Print the number of sources as a comment. */
    const int num_sources = oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES);
    (void) fprintf(file, "# Number of sources: %d\n", num_sources);

    /* Count the number of each type of column. */
    /* Only need to store attribute for first instance of the column type. */
    int num_unique_columns = 0;
    oskar_SkyColumn* unique_columns = 0;
    int* unique_columns_attribute = 0;
    for (c = 0; c < num_columns; ++c)
    {
        const oskar_SkyColumn column_type = oskar_sky_column_type(sky, c);
        const int column_attribute = oskar_sky_column_attribute(sky, c);
        int matched = 0;
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
            unique_columns_attribute[num_unique_columns - 1] = column_attribute;
        }
    }

    /* Print the format string to identify the columns. */
    (void) fprintf(file, "# (");
    if (write_name) (void) fprintf(file, "Name, ");
    if (write_type) (void) fprintf(file, "Type, ");
    for (c = 0; c < num_unique_columns; ++c)
    {
        const oskar_SkyColumn column_type = unique_columns[c];
        if (column_type == OSKAR_SKY_RA_RAD && use_degree_coord_column)
        {
            (void) fprintf(
                    file, "%s", oskar_sky_column_type_to_name(OSKAR_SKY_RA_DEG)
            );
        }
        else if (column_type == OSKAR_SKY_DEC_RAD && use_degree_coord_column)
        {
            (void) fprintf(
                    file, "%s", oskar_sky_column_type_to_name(OSKAR_SKY_DEC_DEG)
            );
        }
        else
        {
            (void) fprintf(
                    file, "%s", oskar_sky_column_type_to_name(column_type)
            );
        }
        if (c < num_unique_columns - 1) fprintf(file, ", ");
    }
    (void) fprintf(file, ") = format\n");

    /* Get handle to columns to check for extended sources. */
    const oskar_Mem* maj = oskar_sky_column_const(sky, OSKAR_SKY_MAJOR_RAD, 0);
    const oskar_Mem* min = oskar_sky_column_const(sky, OSKAR_SKY_MINOR_RAD, 0);

    /* Loop over rows. */
    for (r = 0; r < num_sources; ++r)
    {
        if (write_name)
        {
            fprintf(file, "s%d,", r + 1);
        }
        if (write_type)
        {
            /* Check to see if the source is point or Gaussian. */
            if (maj && min)
            {
                const double ma = oskar_mem_get_element(maj, r, status);
                const double mi = oskar_mem_get_element(min, r, status);
                fprintf(file, (ma > 0.0 && mi > 0.0) ? "GAUSSIAN," : "POINT,");
            }
            else
            {
                fprintf(file, "POINT,");
            }
        }

        /* Loop over unique columns. */
        for (c = 0; c < num_unique_columns; ++c)
        {
            const oskar_SkyColumn column_type = unique_columns[c];
            const int num_columns_of_type = oskar_sky_num_columns_of_type(
                    sky, column_type
            );

            /* If there are multiple columns of this type, print as a vector. */
            if (num_columns_of_type > 1)
            {
                int m = 0; /* How many have we done? */
                (void) fprintf(file, "[");
                for (d = 0; d < num_columns; ++d)
                {
                    if (oskar_sky_column_type(sky, d) == column_type)
                    {
                        const int column_attribute = oskar_sky_column_attribute(
                                sky, d
                        );
                        print_value(
                                sky, file, column_type, column_attribute,
                                use_degree_coord_column, r
                        );
                        if (m < num_columns_of_type - 1)
                        {
                            (void) fprintf(file, ",");
                        }
                        m++;
                    }
                }
                (void) fprintf(file, "]");
            }
            else
            {
                const int column_attribute = unique_columns_attribute[c];
                print_value(
                        sky, file, column_type, column_attribute,
                        use_degree_coord_column, r
                );
            }
            if (c < num_unique_columns - 1)
            {
                (void) fprintf(file, ",");
            }
        }
        (void) fprintf(file, "\n");
    }

    /* Clean up. */
    free(unique_columns);
    free(unique_columns_attribute);
    (void) fclose(file);
}

#ifdef __cplusplus
}
#endif
