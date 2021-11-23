/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_geodetic_spherical_to_ecef.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_geodetic_spherical_to_ecef(int num_points,
        const double* lon_rad, const double* lat_rad, const double* alt_metres,
        double* x, double* y, double* z)
{
    const double a = 6378137.000; /* Equatorial radius (semi-major axis). */
    const double b = 6356752.314; /* Polar radius (semi-minor axis). */
    const double e2 = 1.0 - (b*b) / (a*a);
    int i = 0;
    for (i = 0; i < num_points; ++i)
    {
        const double lat_ = lat_rad[i];
        const double lon_ = lon_rad[i];
        const double alt_ = alt_metres[i];
        const double sin_lat = sin(lat_);
        const double cos_lat = cos(lat_);
        const double sin_lon = sin(lon_);
        const double cos_lon = cos(lon_);
        const double n_phi = a / sqrt(1.0 - e2 * sin_lat * sin_lat);
        x[i] = (n_phi + alt_) * cos_lat * cos_lon;
        y[i] = (n_phi + alt_) * cos_lat * sin_lon;
        z[i] = ((1.0 - e2) * n_phi + alt_) * sin_lat;
    }
}

#ifdef __cplusplus
}
#endif
