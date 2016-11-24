/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#include <gtest/gtest.h>

#include "convert/oskar_convert_ecef_to_geodetic_spherical.h"
#include "convert/oskar_convert_geodetic_spherical_to_ecef.h"
#include "math/oskar_cmath.h"

TEST(station_coord_transforms, geocentric_cartesian_to_geodetic_spherical)
{
    double lon1[] = {0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0,
            120.0, 135.0, 150.0, 165.0, 180.0, -15.0, -30.0, -45.0,
            -60.0, -75.0, -90.0};
    double lat1[] = {-90.0, -80.0, -70.0, -60.0, -50.0, -40.0, -30.0, -20.0,
            -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0};
    double alt1[] = {0.0, 5.0, 10.0, 50.0, 100.0, 200.0, 400.0, 500.0, 340.0,
            84.5, 63.8, 34.8, 73.6, 67.4, 98.3, 12.4, 64.7, 88.6, 224.5};

    const int n = sizeof(lon1) / sizeof(double);
    for (int i = 0; i < n; ++i)
    {
        lon1[i] *= M_PI / 180;
        lat1[i] *= M_PI / 180;
    }

    double x[n], y[n], z[n], lon2[n], lat2[n], alt2[n];
    oskar_convert_geodetic_spherical_to_ecef(n, lon1, lat1, alt1, x, y, z);
    oskar_convert_ecef_to_geodetic_spherical(n, x, y, z, lon2, lat2, alt2);

    for (int i = 0; i < n; ++i)
    {
        EXPECT_NEAR(lon1[i], lon2[i], 1e-8);
        EXPECT_NEAR(lat1[i], lat2[i], 1e-8);
        EXPECT_NEAR(alt1[i], alt2[i], 1e-8);
    }
}
