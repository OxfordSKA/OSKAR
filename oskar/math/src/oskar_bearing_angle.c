/*
 * Copyright (c) 2014-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_bearing_angle.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

double oskar_bearing_angle(double lon1_rad, double lon2_rad,
        double lat1_rad, double lat2_rad)
{
    double delta_lon = 0.0;
    /* http://www.movable-type.co.uk/scripts/latlong.html */
    delta_lon = lon2_rad - lon1_rad;
    return atan2(sin(delta_lon) * cos(lat2_rad),
        cos(lat1_rad) * sin(lat2_rad) -
        sin(lat1_rad) * cos(lat2_rad) * cos(delta_lon));
}

#ifdef __cplusplus
}
#endif
