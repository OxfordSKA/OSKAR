/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_ecef_to_enu.h"
#include "convert/oskar_convert_geodetic_spherical_to_ecef.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_ecef_to_enu(int num_points, const double* ecef_x,
        const double* ecef_y, const double* ecef_z, double lon_rad,
        double lat_rad, double alt_metres, double* x, double* y, double* z)
{
    int i = 0;
    double x0 = 0.0, y0 = 0.0, z0 = 0.0, a = 0.0, b = 0.0, c = 0.0, d = 0.0;
    double sin_lon = 0.0, cos_lon = 0.0, sin_lat = 0.0, cos_lat = 0.0;

    /* Get ECEF coordinates of reference position. */
    oskar_convert_geodetic_spherical_to_ecef(1,
            &lon_rad, &lat_rad, &alt_metres, &x0, &y0, &z0);

    /* Get rotation matrix elements. */
    sin_lon = sin(lon_rad);
    cos_lon = cos(lon_rad);
    sin_lat = sin(lat_rad);
    cos_lat = cos(lat_rad);
    a = -sin_lat * cos_lon;
    b = -sin_lat * sin_lon;
    c = cos_lat * cos_lon;
    d = cos_lat * sin_lon;

    /* Loop over points. */
    for (i = 0; i < num_points; ++i)
    {
        /* Get deltas from reference point. */
        double dx = 0.0, dy = 0.0, dz = 0.0;
        dx = ecef_x[i] - x0;
        dy = ecef_y[i] - y0;
        dz = ecef_z[i] - z0;

        /* Get horizon coordinates. */
        x[i] = -sin_lon * dx + cos_lon * dy;
        y[i] = a * dx + b * dy + cos_lat * dz;
        z[i] = c * dx + d * dy + sin_lat * dz;
    }
}

#ifdef __cplusplus
}
#endif
