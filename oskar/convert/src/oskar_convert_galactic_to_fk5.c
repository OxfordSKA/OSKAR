/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_galactic_to_fk5.h"
#include "math/oskar_cmath.h"

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
    int i = 0, j = 0;

    for (j = 0; j < num_points; ++j)
    {
        double p[3];  /* Input */
        double p1[3]; /* Output */
        double t = 0.0;

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
    int i = 0, j = 0;

    for (j = 0; j < num_points; ++j)
    {
        double p[3];  /* Input */
        double p1[3]; /* Output */
        double t = 0.0;

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
