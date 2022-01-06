/*
 * Copyright (c) 2013-2022, The OSKAR Developers.
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

void oskar_convert_galactic_to_fk5(int num_points, const double* l,
        const double* b, double* ra, double* dec)
{
    int i = 0;
    for (i = 0; i < num_points; ++i)
    {
        double out_x = 0.0, out_y = 0.0, out_z = 0.0, t = 0.0;

        /* Convert Galactic coordinates to Cartesian vector. */
        t = cos(b[i]);
        const double in_x = cos(l[i]) * t;
        const double in_y = sin(l[i]) * t;
        const double in_z = sin(b[i]);

        /* Rotate to equatorial frame. */
        out_x = in_x * rmat[0][0] + in_y * rmat[1][0] + in_z * rmat[2][0];
        out_y = in_x * rmat[0][1] + in_y * rmat[1][1] + in_z * rmat[2][1];
        out_z = in_x * rmat[0][2] + in_y * rmat[1][2] + in_z * rmat[2][2];

        /* Convert Cartesian vector to equatorial coordinates. */
        ra[i] = atan2(out_y, out_x);
        dec[i] = atan2(out_z, sqrt(out_x * out_x + out_y * out_y));

        /* Check range of RA (0 to 2pi). */
        t = fmod(ra[i], 2.0 * M_PI);
        ra[i] = (t >= 0.0) ? t : t + 2.0 * M_PI;
    }
}

#ifdef __cplusplus
}
#endif
