/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_equation_of_equinoxes_fast.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DEG2RAD 0.0174532925199432957692
#define HOUR2RAD 0.261799387799149436539

double oskar_equation_of_equinoxes_fast(double mjd)
{
    /* Days from J2000.0. */
    const double d = mjd - 51544.5;

    /* Longitude of ascending node of the Moon. */
    const double omega = (125.04 - 0.052954 * d) * DEG2RAD;

    /* Mean Longitude of the Sun. */
    const double L = (280.47 + 0.98565 * d) * DEG2RAD;

    /* eqeq = delta_psi * cos(epsilon). */
    const double delta_psi = -0.000319 * sin(omega) - 0.000024 * sin(2.0 * L);
    const double epsilon = (23.4393 - 0.0000004 * d) * DEG2RAD;

    /* Return equation of equinoxes in radians. */
    const double eqeq = delta_psi * cos(epsilon) * HOUR2RAD;
    return eqeq;
}

#ifdef __cplusplus
}
#endif
