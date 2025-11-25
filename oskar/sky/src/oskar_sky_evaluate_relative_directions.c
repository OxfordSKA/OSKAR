/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"

#include "convert/oskar_convert_lon_lat_to_relative_directions.h"

#ifdef __cplusplus
extern "C" {
#endif


void oskar_sky_evaluate_relative_directions(
        oskar_Sky* sky,
        double ra0_rad,
        double dec0_rad,
        int* status
)
{
    if (*status) return;

    /* Convert coordinates. */
    oskar_convert_lon_lat_to_relative_directions(
            oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES),
            oskar_sky_column_const(sky, OSKAR_SKY_RA_RAD, 0),
            oskar_sky_column_const(sky, OSKAR_SKY_DEC_RAD, 0),
            ra0_rad,
            dec0_rad,
            oskar_sky_column(sky, OSKAR_SKY_SCRATCH_L, 0, status),
            oskar_sky_column(sky, OSKAR_SKY_SCRATCH_M, 0, status),
            oskar_sky_column(sky, OSKAR_SKY_SCRATCH_N, 0, status),
            status
    );

    /* Store the reference position. */
    oskar_sky_set_double(sky, OSKAR_SKY_REF_RA_RAD, ra0_rad);
    oskar_sky_set_double(sky, OSKAR_SKY_REF_DEC_RAD, dec0_rad);
}

#ifdef __cplusplus
}
#endif
