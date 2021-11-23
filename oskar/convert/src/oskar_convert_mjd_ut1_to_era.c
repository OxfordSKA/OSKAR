/*
 * Copyright (c) 2014-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_mjd_ut1_to_era.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TWO_PI 6.283185307179586476925287

double oskar_convert_mjd_ut1_to_era(double mjd_ut1)
{
    double theta = 0.0;

    /* Days from J2000.0. */
    const double d = mjd_ut1 - 51544.5;

    /* Fractional part of MJD, in days. */
    const double day_frac = fmod(d, 1.0);

    /* Calculate Earth rotation angle. */
    theta = fmod((TWO_PI * (day_frac + 0.7790572732640 +
            0.00273781191135448 * d)), TWO_PI);
    if (theta < 0.0) theta += TWO_PI;

    return theta;
}

#ifdef __cplusplus
}
#endif
