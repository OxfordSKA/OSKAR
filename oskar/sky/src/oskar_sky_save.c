/*
 * Copyright (c) 2012-2025, The OSKAR Developers.
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


void oskar_sky_save(const oskar_Sky* sky, const char* filename, int* status)
{
    int c = 0, i = 0;
    FILE* file = 0;
    if (*status) return;

    /* Count the number of columns to write. */
    c = OSKAR_SKY_PA_RAD;
    while (c > 0 && !oskar_sky_column_const(sky, (oskar_SkyColumn) c, 0))
    {
        --c;
    }
    const int num_cols_out = c;

    /* Open the output file. */
    file = fopen(filename, "w");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;                      /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }

    /* Get the number of sources. */
    const int num_sources = oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES);

    /* Print a helpful header. */
    (void) fprintf(file, "# Number of sources: %i\n", num_sources);
    if (num_cols_out >  0) (void) fprintf(file, "# RA (deg)");
    if (num_cols_out >  1) (void) fprintf(file, ", Dec (deg)");
    if (num_cols_out >  2) (void) fprintf(file, ", I (Jy)");
    if (num_cols_out >  3) (void) fprintf(file, ", Q (Jy)");
    if (num_cols_out >  4) (void) fprintf(file, ", U (Jy)");
    if (num_cols_out >  5) (void) fprintf(file, ", V (Jy)");
    if (num_cols_out >  6) (void) fprintf(file, ", Ref. freq. (Hz)");
    if (num_cols_out >  7) (void) fprintf(file, ", Spectral index");
    if (num_cols_out >  8) (void) fprintf(file, ", Rotation measure (rad/m^2)");
    if (num_cols_out >  9) (void) fprintf(file, ", FWHM major (arcsec)");
    if (num_cols_out > 10) (void) fprintf(file, ", FWHM minor (arcsec)");
    if (num_cols_out > 11) (void) fprintf(file, ", Position angle (deg)");
    if (num_cols_out >  0) (void) fprintf(file, "\n");

    /* Print out sky model in ASCII format. */
    for (i = 0; i < num_sources; ++i)
    {
        for (c = 1; c <= num_cols_out; ++c)
        {
            oskar_SkyColumn col_type = (oskar_SkyColumn) c;
            const double value = oskar_sky_data(sky, col_type, 0, i);
            switch (col_type)
            {
            case OSKAR_SKY_RA_RAD:
            case OSKAR_SKY_DEC_RAD:
            case OSKAR_SKY_PA_RAD:
                (void) fprintf(file, "% 17.12f", value * RAD2DEG);
                break;
            case OSKAR_SKY_MAJOR_RAD:
            case OSKAR_SKY_MINOR_RAD:
                (void) fprintf(file, "% 17.12f", value * RAD2ARCSEC);
                break;
            default:
                (void) fprintf(file, "% 20.14e", value);
                break;
            }
            if (c != num_cols_out) (void) fprintf(file, ",");
        }
        if (num_cols_out > 0) (void) fprintf(file, "\n");
    }
    (void) fclose(file);
}

#ifdef __cplusplus
}
#endif
