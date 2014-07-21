/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#include <oskar_convert_galactic_to_fk5.h>
#include <oskar_cmath.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Equatorial to Galactic rotation matrix. */
static const double rmat[3][3] =
    { {-0.054875539726, -0.873437108010, -0.483834985808},
      { 0.494109453312, -0.444829589425,  0.746982251810},
      {-0.867666135858, -0.198076386122,  0.455983795705} };

void oskar_convert_galactic_to_fk5_f(int num_points, const float* l,
        const float* b, float* ra, float* dec)
{
    int i, j;

    for (j = 0; j < num_points; ++j)
    {
        double p[3];  /* Input */
        double p1[3]; /* Output */
        double t;

        /* Convert Galactic coordinates to Cartesian vector. */
        t = cos(b[j]);
        p[0] = cos(l[j]) * t;
        p[1] = sin(l[j]) * t;
        p[2] = sin(b[j]);

        /* Rotate to equatorial frame. */
        for (i = 0; i < 3; i++)
        {
            p1[i] = p[0] * rmat[0][i] + p[1] * rmat[1][i] + p[2] * rmat[2][i];
        }

        /* Convert Cartesian vector to equatorial coordinates. */
        /* RA = atan2(y, x) */
        ra[j] = atan2(p1[1], p1[0]);
        /* DEC = atan2(z, sqrt(x*x + y*y)) */
        dec[j] = atan2(p1[2], sqrt(p1[0]*p1[0] + p1[1]*p1[1]));

        /* Check range of RA (0 to 2pi). */
        t = fmod(ra[j], 2.0 * M_PI);
        ra[j] = (t >= 0.0) ? t : t + 2.0 * M_PI;
    }
}

void oskar_convert_galactic_to_fk5_d(int num_points, const double* l,
        const double* b, double* ra, double* dec)
{
    int i, j;

    for (j = 0; j < num_points; ++j)
    {
        double p[3];  /* Input */
        double p1[3]; /* Output */
        double t;

        /* Convert Galactic coordinates to Cartesian vector. */
        t = cos(b[j]);
        p[0] = cos(l[j]) * t;
        p[1] = sin(l[j]) * t;
        p[2] = sin(b[j]);

        /* Rotate to equatorial frame. */
        for (i = 0; i < 3; i++)
        {
            p1[i] = p[0] * rmat[0][i] + p[1] * rmat[1][i] + p[2] * rmat[2][i];
        }

        /* Convert Cartesian vector to equatorial coordinates. */
        /* RA = atan2(y, x) */
        ra[j] = atan2(p1[1], p1[0]);
        /* DEC = atan2(z, sqrt(x*x + y*y)) */
        dec[j] = atan2(p1[2], sqrt(p1[0]*p1[0] + p1[1]*p1[1]));

        /* Check range of RA (0 to 2pi). */
        t = fmod(ra[j], 2.0 * M_PI);
        ra[j] = (t >= 0.0) ? t : t + 2.0 * M_PI;
    }
}

#ifdef __cplusplus
}
#endif
