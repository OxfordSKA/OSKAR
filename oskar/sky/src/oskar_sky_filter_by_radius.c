/*
 * Copyright (c) 2012-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_angular_distance.h"
#include "math/oskar_cmath.h"
#include "sky/oskar_sky.h"
#include "sky/private_sky.h"

#ifdef __cplusplus
extern "C" {
#endif


void oskar_sky_filter_by_radius(
        oskar_Sky* sky,
        double inner_radius_rad,
        double outer_radius_rad,
        double ra0_rad,
        double dec0_rad,
        int* status
)
{
    if (*status) return;

    /* Return immediately if no filtering should be done. */
    if (inner_radius_rad == 0.0 && outer_radius_rad >= M_PI) return;
    if (outer_radius_rad < inner_radius_rad)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;             /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }

    /* Get the meta-data. */
    const int type = oskar_sky_int(sky, OSKAR_SKY_PRECISION);
    const int location = oskar_sky_int(sky, OSKAR_SKY_MEM_LOCATION);
    const int num_sources = oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES);
    const int num_columns = oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS);

    /* Switch on location and data type. */
    if (location == OSKAR_CPU)
    {
        int c = 0, in = 0, out = 0;
        void* col = oskar_mem_void(sky->ptr_columns);
        const void* ra_  = oskar_mem_void_const(
                oskar_sky_column_const(sky, OSKAR_SKY_RA_RAD, 0)
        );
        const void* dec_ = oskar_mem_void_const(
                oskar_sky_column_const(sky, OSKAR_SKY_DEC_RAD, 0)
        );
        if (type == OSKAR_SINGLE)
        {
            for (in = 0, out = 0; in < num_sources; ++in)
            {
                const double dist = oskar_angular_distance(
                        ((const float*) ra_)[in], ra0_rad,
                        ((const float*) dec_)[in], dec0_rad
                );
                if (!(dist >= inner_radius_rad && dist < outer_radius_rad))
                {
                    continue;
                }
                #pragma GCC unroll 8
                for (c = 0; c < num_columns; ++c)
                {
                    ((float**) col)[c][out] = ((float**) col)[c][in];
                }
                out++;
            }
        }
        else if (type == OSKAR_DOUBLE)
        {
            for (in = 0, out = 0; in < num_sources; ++in)
            {
                const double dist = oskar_angular_distance(
                        ((const double*) ra_)[in], ra0_rad,
                        ((const double*) dec_)[in], dec0_rad
                );
                if (!(dist >= inner_radius_rad && dist < outer_radius_rad))
                {
                    continue;
                }
                #pragma GCC unroll 8
                for (c = 0; c < num_columns; ++c)
                {
                    ((double**) col)[c][out] = ((double**) col)[c][in];
                }
                out++;
            }
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
            return;                                       /* LCOV_EXCL_LINE */
        }

        /* Set the new size of the sky model. */
        oskar_sky_resize(sky, out, status);
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;                 /* LCOV_EXCL_LINE */
    }
}

#ifdef __cplusplus
}
#endif
