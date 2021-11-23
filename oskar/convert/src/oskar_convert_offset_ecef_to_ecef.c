/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_offset_ecef_to_ecef.h"
#include "convert/oskar_convert_geodetic_spherical_to_ecef.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_offset_ecef_to_ecef(int num_points, const double* offset_x,
        const double* offset_y, const double* offset_z, double lon_rad,
        double lat_rad, double alt_metres, double* x, double* y, double* z)
{
    /* Compute ECEF coordinates of reference point. */
    double x_r = 0.0, y_r = 0.0, z_r = 0.0;
    int i = 0;
    oskar_convert_geodetic_spherical_to_ecef(1, &lon_rad, &lat_rad,
            &alt_metres, &x_r, &y_r, &z_r);

    /* Add on the coordinates of the reference point. */
    for (i = 0; i < num_points; ++i)
    {
        x[i] = offset_x[i] + x_r;
        y[i] = offset_y[i] + y_r;
        z[i] = offset_z[i] + z_r;
    }
}

#ifdef __cplusplus
}
#endif
