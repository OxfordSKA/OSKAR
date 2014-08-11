/*
 * Copyright (c) 2014, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef OSKAR_CONVERT_ECEF_TO_GEODETIC_SPHERICAL_INLINE_H_
#define OSKAR_CONVERT_ECEF_TO_GEODETIC_SPHERICAL_INLINE_H_

/**
 * @file oskar_convert_ecef_to_geodetic_spherical_inline.h
 */

#include <oskar_global.h>
#include <oskar_cmath.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ECEF_TO_GEODETIC_SPHERICAL_METHOD 1

/* Single precision. */
OSKAR_INLINE
void oskar_convert_ecef_to_geodetic_spherical_inline_f(const float x,
        const float y, const float z, float* lon, float* lat, float* alt)
{
    const float a = 6378137.000f; /* Equatorial radius (semi-major axis). */
    const float b = 6356752.314f; /* Polar radius (semi-minor axis). */
#if ECEF_TO_GEODETIC_SPHERICAL_METHOD == 1
    /* Vermeille (2002) method: */
    /* TODO Maybe try an iterative approach for single precision? */
    const float ba = b / a;
    const float e2 = 1.0f - ba * ba;
    const float e4 = e2 * e2;
    float p, q, r, s, t, u, v, w, k, D, L2, L, M;
    L2 = x*x + y*y;
    L = sqrtf(L2);
    p = x / a;
    q = y / a;
    p = p * p + q * q; /* (X^2 + Y^2) / a^2 */
    r = z / a;
    q = ba * ba * r * r; /* (1 - e^2) * (Z^2 / a^2) */
    r = (p + q - e4) / 6.0f;
    s = (e4 * p * q) / (4.0f * r*r*r);
    t = powf(1.0f + s + sqrtf(s * (2.0f + s)), 1.0f/3.0f);
    u = r * (1.0f + t + 1.0f / t);
    v = sqrtf(u*u + e4 * q);
    w = e2 * (u + v - q) / (2.0f * v);
    k = sqrtf(u + v + w * w) - w;
    D = k * L / (k + e2);
    M = sqrtf(D*D + z*z);
    *lon = atan2f(y, x);
    *lat = 2.0f * atan2f(z, D + M);
    *alt = M * (k - ba * ba) / k;
#else
    /* Borkowski (1989) method: */
    float r, b_sgn, phi = 0.0f, h = 0.0f;
    r = sqrtf(x*x + y*y);
    b_sgn = (z < 0.0f) ? -b : b;
    if (z != 0.0f && r != 0.0f)
    {
        float ab, ar, bz, E, F, P, Q, D, v, G, t;
        ab = (a + b) * (a - b); /* a^2 - b^2 */
        ar = 1.0f / (a * r);
        bz = b_sgn * z;
        E = ar * (bz - ab);
        F = ar * (bz + ab);
        P = (4.0f/3.0f) * (E * F + 1.0f);
        Q = 2.0f * (E + F) * (E - F); /* 2 * (E^2 - F^2) */
        D = P*P*P + Q*Q;
        if (D < 0.0f)
        {
            v = 2.0f * sqrtf(-P) * cosf((1.0f/3.0f) *
                    acosf((Q/P) * powf(-P, -0.5f)));
        }
        else
        {
            float Ds;
            Ds = sqrtf(D);
            v = powf(Ds - Q, 1.0f/3.0f) - powf(Ds + Q, 1.0f/3.0f);
        }
        G = 0.5f * (sqrtf(E*E + v) + E);
        t = sqrtf(G*G + (F - v * G) / (2.0f * G - E)) - G;

        /* Evaluate geodetic latitude and altitude. */
        phi = atanf(a * (1.0f + t) * (1.0f - t) / (2.0f * b_sgn * t));
        h = (r - a * t) * cosf(phi) + (z - b_sgn) * sinf(phi);
    }
    else if (z == 0.0f)
    {
        phi = 0.0f;
        h = r - a;
    }
    else if (r == 0.0f)
    {
        phi = (z > 0.0f) ? (float) M_PI_2 : (float) -M_PI_2;
        h = fabsf(z) - b;
    }

    /* Store results. */
    *lon = atan2f(y, x);
    *lat = phi;
    *alt = h;
#endif
}

/* Double precision. */
OSKAR_INLINE
void oskar_convert_ecef_to_geodetic_spherical_inline_d(const double x,
        const double y, const double z, double* lon, double* lat, double* alt)
{
    const double a = 6378137.000; /* Equatorial radius (semi-major axis). */
    const double b = 6356752.314; /* Polar radius (semi-minor axis). */
#if ECEF_TO_GEODETIC_SPHERICAL_METHOD == 1
    /* Vermeille (2002) method: */
    const double ba = b / a;
    const double e2 = 1.0 - ba * ba;
    const double e4 = e2 * e2;
    double p, q, r, s, t, u, v, w, k, D, L2, L, M;
    L2 = x*x + y*y;
    L = sqrt(L2);
    p = x / a;
    q = y / a;
    p = p * p + q * q; /* (X^2 + Y^2) / a^2 */
    r = z / a;
    q = ba * ba * r * r; /* (1 - e^2) * (Z^2 / a^2) */
    r = (p + q - e4) / 6.0;
    s = (e4 * p * q) / (4.0 * r*r*r);
    t = pow(1.0 + s + sqrt(s * (2.0 + s)), 1.0/3.0);
    u = r * (1.0 + t + 1.0 / t);
    v = sqrt(u*u + e4 * q);
    w = e2 * (u + v - q) / (2.0 * v);
    k = sqrt(u + v + w * w) - w;
    D = k * L / (k + e2);
    M = sqrt(D*D + z*z);
    *lon = atan2(y, x);
    *lat = 2.0 * atan2(z, D + M);
    *alt = M * (k - ba * ba) / k;
#else
    /* Borkowski (1989) method: */
    double r, b_sgn, phi = 0.0, h = 0.0;
    r = sqrt(x*x + y*y);
    b_sgn = (z < 0.0) ? -b : b;
    if (z != 0.0 && r != 0.0)
    {
        double ab, ar, bz, E, F, P, Q, D, v, G, t;
        ab = (a + b) * (a - b); /* a^2 - b^2 */
        ar = 1.0 / (a * r);
        bz = b_sgn * z;
        E = ar * (bz - ab);
        F = ar * (bz + ab);
        P = (4.0/3.0) * (E * F + 1.0);
        Q = 2.0 * (E + F) * (E - F); /* 2 * (E^2 - F^2) */
        D = P*P*P + Q*Q;
        if (D < 0.0)
        {
            v = 2.0 * sqrt(-P) * cos((1.0/3.0) *
                    acos((Q/P) * pow(-P, -0.5)));
        }
        else
        {
            double Ds;
            Ds = sqrt(D);
            v = pow(Ds - Q, 1.0/3.0) - pow(Ds + Q, 1.0/3.0);
        }
        G = 0.5 * (sqrt(E*E + v) + E);
        t = sqrt(G*G + (F - v * G) / (2.0 * G - E)) - G;

        /* Evaluate geodetic latitude and altitude. */
        phi = atan(a * (1.0 + t) * (1.0 - t) / (2.0 * b_sgn * t));
        h = (r - a * t) * cos(phi) + (z - b_sgn) * sin(phi);
    }
    else if (z == 0.0)
    {
        phi = 0.0;
        h = r - a;
    }
    else if (r == 0.0)
    {
        phi = (z > 0.0) ? M_PI_2 : -M_PI_2;
        h = fabs(z) - b;
    }

    /* Store results. */
    *lon = atan2(y, x);
    *lat = phi;
    *alt = h;
#endif
}

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_ECEF_TO_GEODETIC_SPHERICAL_INLINE_H_ */
