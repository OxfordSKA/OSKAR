/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_mjd_to_gmst.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Seconds to radians. */
#define SEC2RAD 7.2722052166430399038487e-5

#ifndef M_2PI
#define M_2PI 6.28318530717958647693
#endif

double oskar_convert_mjd_to_gmst(double mjd)
{
    /* Days from J2000.0. */
    const double d = mjd - 51544.5;

    /* Centuries from J2000.0. */
    const double t = d / 36525.0;

    /* GMST at this time. */
    const double gmst = fmod(mjd, 1.0) * M_2PI +
            (24110.54841 + (8640184.812866 +
            (0.093104 - 6.2e-6 * t) * t) * t) * SEC2RAD;

    /* Range check (0 to 2pi). */
    const double x = fmod(gmst, M_2PI);
    return (x >= 0.0) ? x : x + M_2PI;
}

#ifdef __cplusplus
}
#endif
