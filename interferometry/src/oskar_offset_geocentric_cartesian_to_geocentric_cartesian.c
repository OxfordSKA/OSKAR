/*
 * Copyright (c) 2011, The University of Oxford
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

#include "interferometry/oskar_offset_geocentric_cartesian_to_geocentric_cartesian.h"
#include "interferometry/oskar_geodetic_spherical_to_geocentric_cartesian.h"

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_offset_geocentric_cartesian_to_geocentric_cartesian(int n,
        const double* x_offset, const double* y_offset,
        const double* z_offset, double longitude, double latitude,
        double altitude, double* x, double* y, double* z)
{
    /* Compute ECEF coordinates of reference point. */
    double x_r = 0.0, y_r = 0.0, z_r = 0.0;
    int i;
    oskar_geodetic_spherical_to_geocentric_cartesian(1, &longitude, &latitude,
            &altitude, &x_r, &y_r, &z_r);

    /* Add on the coordinates of the reference point. */
    for (i = 0; i < n; ++i)
    {
        x[i] = x_offset[i] + x_r;
        y[i] = y_offset[i] + y_r;
        z[i] = z_offset[i] + z_r;
    }
}

#ifdef __cplusplus
}
#endif
