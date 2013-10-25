/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include <oskar_linspace.h>
#include <oskar_convert_tangent_plane_direction_to_lon_lat.h>
#include <oskar_convert_lon_lat_to_tangent_plane_direction.h>
#include <oskar_evaluate_image_lm_grid.h>

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

TEST(sph_lm, test)
{
    // Image size.
    int num_l = 10;
    int num_m = 10;
    double fov_lon_deg = 10.0;
    double fov_lat_deg = 10.0;

    // Set up the reference point.
    double lon0 = 10.0 * M_PI / 180.0;
    double lat0 = 50.0 * M_PI / 180.0;

    // Set up the grid.
    int num_points = num_l * num_m;
    std::vector<double> grid_l(num_points), grid_m(num_points);
    std::vector<double> grid_RA(num_points), grid_Dec(num_points);
    oskar_evaluate_image_lm_grid_d(num_l, num_m, fov_lon_deg * M_PI / 180.0,
            fov_lat_deg * M_PI / 180.0, &grid_l[0], &grid_m[0]);

    // Convert from l,m grid to spherical coordinates.
    oskar_convert_tangent_plane_direction_to_lon_lat_d(num_points, lon0, lat0,
            &grid_l[0], &grid_m[0], &grid_RA[0], &grid_Dec[0]);

    // Check reverse direction.
    std::vector<double> temp_l(num_points), temp_m(num_points);
    oskar_convert_lon_lat_to_tangent_plane_direction_d(num_points, lon0, lat0,
            &grid_RA[0], &grid_Dec[0], &temp_l[0], &temp_m[0]);

    for (int i = 0; i < num_points; ++i)
    {
        EXPECT_NEAR(grid_l[i], temp_l[i], 1e-15);
        EXPECT_NEAR(grid_m[i], temp_m[i], 1e-15);
    }
}
