/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <math.h>
#include "convert/oskar_convert_ecef_to_geodetic_spherical.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_ecef_to_geodetic_spherical(int num_points,
        const double* x, const double* y, const double* z,
        double* lon_rad, double* lat_rad, double* alt_m)
{
    int i = 0;
    const double a = 6378137.000; /* Equatorial radius (semi-major axis). */
    const double b = 6356752.314; /* Polar radius (semi-minor axis). */
    const double ba = b / a;
    const double e2 = 1.0 - ba * ba;
    const double e4 = e2 * e2;
    for (i = 0; i < num_points; ++i)
    {
        double p = 0.0, q = 0.0, r = 0.0, s = 0.0, t = 0.0;
        double u = 0.0, v = 0.0, w = 0.0, k = 0.0;
        double D = 0.0, L2 = 0.0, L = 0.0, M = 0.0;
        L2 = x[i]*x[i] + y[i]*y[i];
        L = sqrt(L2);
        p = x[i] / a;
        q = y[i] / a;
        p = p * p + q * q; /* (X^2 + Y^2) / a^2 */
        r = z[i] / a;
        q = ba * ba * r * r; /* (1 - e^2) * (Z^2 / a^2) */
        r = (p + q - e4) / 6.0;
        s = (e4 * p * q) / (4.0 * r*r*r);
        t = pow(1.0 + s + sqrt(s * (2.0 + s)), 1.0/3.0);
        u = r * (1.0 + t + 1.0 / t);
        v = sqrt(u*u + e4 * q);
        w = e2 * (u + v - q) / (2.0 * v);
        k = sqrt(u + v + w * w) - w;
        D = k * L / (k + e2);
        M = sqrt(D*D + z[i]*z[i]);
        lon_rad[i] = atan2(y[i], x[i]);
        lat_rad[i] = 2.0 * atan2(z[i], D + M);
        alt_m[i] = M * (k - ba * ba) / k;
    }
}

#ifdef __cplusplus
}
#endif
