/*
 * Copyright (c) 2013, The University of Oxford
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

#include <oskar_convert_ecef_to_geodetic_spherical.h>
#include <math.h>

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923132169163975144 /* pi/2 */
#endif

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_ecef_to_geodetic_spherical(int n, const double* x,
        const double* y, const double* z, double* lon, double* lat, double* alt)
{
    int i;
    const double a = 6378137.000; /* Equatorial radius (semi-major axis). */
    const double b_unsigned = 6356752.314; /* Polar radius (semi-minor axis). */
    for (i = 0; i < n; ++i)
    {
        double X, Y, Z, r, b, E, F, P, Q, D, Ds, v, G, t, phi, h;
        X = x[i];
        Y = y[i];
        Z = z[i];
        r = sqrt(X*X + Y*Y);
        b = (Z < 0.0) ? -b_unsigned : b_unsigned;
        phi = 0.0;
        h = 0.0;
        if (Z != 0.0 && r != 0.0)
        {
            E = (b * Z - (a*a - b*b)) / (a * r);
            F = (b * Z + (a*a - b*b)) / (a * r);
            P = (4.0/3.0) * (E * F + 1.0);
            Q = 2.0 * (E*E - F*F);
            D = P*P*P + Q*Q;
            if (D < 0.0)
            {
                v = 2.0 * sqrt(-P) *
                        cos((1.0/3.0) * acos((Q/P) * pow(-P, -0.5)));
            }
            else
            {
                Ds = sqrt(D);
                v = pow(Ds - Q, 1.0/3.0) - pow(Ds + Q, 1.0/3.0);
            }
            G = (sqrt(E*E + v) + E) / 2.0;
            t = sqrt(G*G + (F - v * G) / (2.0 * G - E)) - G;

            /* Evaluate geodetic latitude and altitude. */
            phi = atan(a * (1 - t*t) / (2.0 * b * t));
            h = (r - a * t) * cos(phi) + (Z - b) * sin(phi);
        }
        else if (Z == 0.0)
        {
            phi = 0.0;
            h = r - a;
        }
        else if (r == 0.0)
        {
            phi = (Z > 0.0) ? M_PI_2 : -M_PI_2;
            h = fabs(Z) - b_unsigned;
        }

        /* Store results. */
        lon[i] = atan2(Y, X);
        lat[i] = phi;
        alt[i] = h;
    }
}


#ifdef __cplusplus
}
#endif
