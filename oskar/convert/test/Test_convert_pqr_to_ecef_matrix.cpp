/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "convert/oskar_convert_pqr_to_ecef_matrix.h"
#include "convert/oskar_convert_geodetic_spherical_to_ecef.h"
#include "math/oskar_cmath.h"

#include <cstdlib>
#include <cstdio>

static void transpose(double m[9])
{
    double t = 0.0;
    t = m[1]; m[1] = m[3]; m[3] = t;
    t = m[2]; m[2] = m[6]; m[6] = t;
    t = m[5]; m[5] = m[7]; m[7] = t;
}

TEST(convert_pqr_to_ecef_matrix, station_s8_1)
{
    // Check conversion matrix for station S8-1.
    // This is the transpose of the answer we need to get.
    double pqr_to_ecef_for_s8_1[9] = {
             0.09388869,  0.52638352,  0.84504752,
             0.91113548,  0.29667722, -0.2860328,
            -0.4012693,   0.79680802, -0.45175206
    };
    // Transpose because the matrix as written above is actually
    // ecef_to_pqr, and we want it the other way for this test.
    // HOWEVER, the thing that should be written to the PHASED_ARRAY
    // table really is ecef_to_pqr (the values in the order above)!
    transpose(pqr_to_ecef_for_s8_1);
    // Now pqr_to_ecef_for_s8_1 is what it says it is.

    // Input values for S8-1.
    double rotation_angle_rad = 251.3 * M_PI / 180;
    double station_coords_wgs84[3] = {
            116.729640800, -26.856150266, 330.104
    };

    // Convert WGS84 values to ECEF for function call.
    double station_coords_ecef[3] = {0., 0., 0.};
    station_coords_wgs84[0] *= M_PI / 180.0;
    station_coords_wgs84[1] *= M_PI / 180.0;
    oskar_convert_geodetic_spherical_to_ecef(
            1,
            &station_coords_wgs84[0],
            &station_coords_wgs84[1],
            &station_coords_wgs84[2],
            &station_coords_ecef[0],
            &station_coords_ecef[1],
            &station_coords_ecef[2]
    );

    // Call the function to test it.
    double pqr_to_ecef[9];
    oskar_convert_pqr_to_ecef_matrix(
            station_coords_ecef, rotation_angle_rad, pqr_to_ecef
    );

    // Compare with known result.
    for (int i = 0; i < 9; ++i)
    {
        EXPECT_NEAR(pqr_to_ecef[i], pqr_to_ecef_for_s8_1[i], 1e-7);
    }
    // const char* format = "%.4f, %.4f, %.4f\n";
    // printf(format, pqr_to_ecef[0], pqr_to_ecef[1], pqr_to_ecef[2]);
    // printf(format, pqr_to_ecef[3], pqr_to_ecef[4], pqr_to_ecef[5]);
    // printf(format, pqr_to_ecef[6], pqr_to_ecef[7], pqr_to_ecef[8]);
}
