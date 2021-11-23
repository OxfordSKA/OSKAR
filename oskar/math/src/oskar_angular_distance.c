/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_angular_distance.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

double oskar_angular_distance(double lon1_rad, double lon2_rad,
        double lat1_rad, double lat2_rad)
{
    double sin_delta_lat = 0.0, sin_delta_lon = 0.0;
    sin_delta_lat = sin(0.5 * (lat2_rad - lat1_rad));
    sin_delta_lon = sin(0.5 * (lon2_rad - lon1_rad));
    return 2.0 * asin( sqrt(sin_delta_lat*sin_delta_lat +
            cos(lat1_rad) * cos(lat2_rad) * sin_delta_lon*sin_delta_lon) );
}

#ifdef __cplusplus
}
#endif
