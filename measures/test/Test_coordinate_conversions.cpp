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
#include <oskar_convert_apparent_ra_dec_to_relative_direction_cosines.h>
#include <oskar_convert_relative_direction_cosines_to_apparent_ra_dec.h>
#include <oskar_convert_relative_direction_cosines_to_enu_direction_cosines.h>
#include <oskar_convert_lon_lat_to_xyz.h>
#include <oskar_convert_xyz_to_lon_lat.h>
#include <oskar_evaluate_image_lm_grid.h>
#include <oskar_get_error_string.h>

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define D2R M_PI/180.0

TEST(coordinate_conversions, lon_lat_to_xyz)
{
    int num_pts = 1;
    std::vector<double> x(num_pts), y(num_pts), z(num_pts);
    std::vector<double> lon_in(num_pts), lat_in(num_pts);
    std::vector<double> lon_out(num_pts), lat_out(num_pts);
    double delta = 1e-8;

    lon_in[0] = 50.0 * M_PI/180.0;
    lat_in[0] = 30.0 * M_PI/180.0;

    oskar_convert_lon_lat_to_xyz_d(num_pts, &x[0], &y[0], &z[0], &lon_in[0],
            &lat_in[0]);
    oskar_convert_xyz_to_lon_lat_d(num_pts, &lon_out[0], &lat_out[0], &x[0],
            &y[0], &z[0]);

    ASSERT_NEAR(lon_in[0], lon_out[0], delta);
    ASSERT_NEAR(lat_in[0], lat_out[0], delta);
}

TEST(coordinate_conversions, ra_dec_to_direction_cosines)
{
    using std::vector;

    // Image size.
    int num_l = 10;
    int num_m = 10;
    double fov_lon_deg = 10.0;
    double fov_lat_deg = 10.0;

    // Set up the reference point.
    double ra0 = 10.0 * M_PI / 180.0;
    double dec0 = 50.0 * M_PI / 180.0;

    // Set up the grid.
    int num_points = num_l * num_m;
    vector<double> l_1(num_points), m_1(num_points);
    vector<double> ra(num_points), dec(num_points);
    oskar_evaluate_image_lm_grid_d(num_l, num_m, fov_lon_deg * M_PI / 180.0,
            fov_lat_deg * M_PI / 180.0, &l_1[0], &m_1[0]);

    // Convert from l,m grid to spherical coordinates.
    oskar_convert_relative_direction_cosines_to_apparent_ra_dec_d(num_points, ra0, dec0,
            &l_1[0], &m_1[0], &ra[0], &dec[0]);

    // Check reverse direction.
    vector<double> l_2(num_points), m_2(num_points), n_2(num_points);
    oskar_convert_apparent_ra_dec_to_relative_direction_cosines_d(num_points,
            &ra[0], &dec[0], ra0, dec0, &l_2[0], &m_2[0], &n_2[0]);

    for (int i = 0; i < num_points; ++i)
    {
        ASSERT_NEAR(l_1[i], l_2[i], 1e-15);
        ASSERT_NEAR(m_1[i], m_2[i], 1e-15);
        double n_1 = sqrt(1.0 - l_1[i]*l_1[i] - m_1[i]*m_1[i]);
        ASSERT_NEAR(n_1, n_2[i], 1e-15);
    }
}


TEST(coordinate_conversions, relative_direction_cosines_to_enu_direction_cosines)
{
    int type, loc, num = 1, status = 0;
    double lat, lst, ra0, ha0, dec0, tol;

    // Observer's location and LST.
    lat = 73 * D2R;
    lst = 12 * D2R;

    // RA and Dec of phase centre.
    ra0  = 36 * D2R;
    dec0 = 60 * D2R;
    ha0  = lst - ra0;

    // Single precision.
    {
        oskar_Mem *ra, *dec, *l, *m, *n, *x, *y, *z, *x_gpu, *y_gpu, *z_gpu;
        type = OSKAR_SINGLE;
        loc = OSKAR_LOCATION_CPU;
        tol = 1e-5;

        ra = oskar_mem_create(type, loc, num, &status);
        dec = oskar_mem_create(type, loc, num, &status);
        l = oskar_mem_create(type, loc, num, &status);
        m = oskar_mem_create(type, loc, num, &status);
        n = oskar_mem_create(type, loc, num, &status);
        x = oskar_mem_create(type, loc, num, &status);
        y = oskar_mem_create(type, loc, num, &status);
        z = oskar_mem_create(type, loc, num, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Set RA and Dec of point.
        oskar_mem_set_value_real(ra,  40 * D2R, 0, 0, &status);
        oskar_mem_set_value_real(dec, 65 * D2R, 0, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute l, m, n directions.
        oskar_convert_apparent_ra_dec_to_relative_direction_cosines(num,
                ra, dec, ra0, dec0, l, m, n, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute ENU directions.
        oskar_convert_relative_direction_cosines_to_enu_direction_cosines(
                x, y, z, num, l, m, n, ha0, dec0, lat, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check values.
        ASSERT_NEAR(oskar_mem_float(x, &status)[0],
                0.198407255801263, tol);
        ASSERT_NEAR(oskar_mem_float(y, &status)[0],
                -0.0918661536386987, tol);
        ASSERT_NEAR(oskar_mem_float(z, &status)[0],
                0.97580488349928, tol);

        // Free input data.
        oskar_mem_free(ra, &status);
        oskar_mem_free(dec, &status);
        oskar_mem_free(l, &status);
        oskar_mem_free(m, &status);
        oskar_mem_free(n, &status);
        free(ra); // FIXME Remove after updating oskar_mem_free().
        free(dec); // FIXME Remove after updating oskar_mem_free().
        free(l); // FIXME Remove after updating oskar_mem_free().
        free(m); // FIXME Remove after updating oskar_mem_free().
        free(n); // FIXME Remove after updating oskar_mem_free().

        // GPU version.
        loc = OSKAR_LOCATION_GPU;

        ra = oskar_mem_create(type, loc, num, &status);
        dec = oskar_mem_create(type, loc, num, &status);
        l = oskar_mem_create(type, loc, num, &status);
        m = oskar_mem_create(type, loc, num, &status);
        n = oskar_mem_create(type, loc, num, &status);
        x_gpu = oskar_mem_create(type, loc, num, &status);
        y_gpu = oskar_mem_create(type, loc, num, &status);
        z_gpu = oskar_mem_create(type, loc, num, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Set RA and Dec of point.
        oskar_mem_set_value_real(ra,  40 * D2R, 0, 0, &status);
        oskar_mem_set_value_real(dec, 65 * D2R, 0, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute l, m, n directions.
        oskar_convert_apparent_ra_dec_to_relative_direction_cosines(num,
                ra, dec, ra0, dec0, l, m, n, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute ENU directions.
        oskar_convert_relative_direction_cosines_to_enu_direction_cosines(
                x_gpu, y_gpu, z_gpu, num, l, m, n, ha0, dec0, lat, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check values.
        double max_err, mean_err;
        oskar_mem_evaluate_relative_error(x_gpu, x,
                0, &max_err, &mean_err, 0, &status);
        ASSERT_LT(max_err, tol);
        ASSERT_LT(mean_err, tol);
        oskar_mem_evaluate_relative_error(y_gpu, y,
                0, &max_err, &mean_err, 0, &status);
        ASSERT_LT(max_err, tol);
        ASSERT_LT(mean_err, tol);
        oskar_mem_evaluate_relative_error(z_gpu, z,
                0, &max_err, &mean_err, 0, &status);
        ASSERT_LT(max_err, tol);
        ASSERT_LT(mean_err, tol);

        // Free input data.
        oskar_mem_free(ra, &status);
        oskar_mem_free(dec, &status);
        oskar_mem_free(l, &status);
        oskar_mem_free(m, &status);
        oskar_mem_free(n, &status);
        free(ra); // FIXME Remove after updating oskar_mem_free().
        free(dec); // FIXME Remove after updating oskar_mem_free().
        free(l); // FIXME Remove after updating oskar_mem_free().
        free(m); // FIXME Remove after updating oskar_mem_free().
        free(n); // FIXME Remove after updating oskar_mem_free().

        // Free xyz directions.
        oskar_mem_free(x, &status);
        oskar_mem_free(y, &status);
        oskar_mem_free(z, &status);
        oskar_mem_free(x_gpu, &status);
        oskar_mem_free(y_gpu, &status);
        oskar_mem_free(z_gpu, &status);
        free(x); // FIXME Remove after updating oskar_mem_free().
        free(y); // FIXME Remove after updating oskar_mem_free().
        free(z); // FIXME Remove after updating oskar_mem_free().
        free(x_gpu); // FIXME Remove after updating oskar_mem_free().
        free(y_gpu); // FIXME Remove after updating oskar_mem_free().
        free(z_gpu); // FIXME Remove after updating oskar_mem_free().
    }

    // Double precision.
    {
        oskar_Mem *ra, *dec, *l, *m, *n, *x, *y, *z, *x_gpu, *y_gpu, *z_gpu;
        type = OSKAR_DOUBLE;
        loc = OSKAR_LOCATION_CPU;
        tol = 1e-12;

        ra = oskar_mem_create(type, loc, num, &status);
        dec = oskar_mem_create(type, loc, num, &status);
        l = oskar_mem_create(type, loc, num, &status);
        m = oskar_mem_create(type, loc, num, &status);
        n = oskar_mem_create(type, loc, num, &status);
        x = oskar_mem_create(type, loc, num, &status);
        y = oskar_mem_create(type, loc, num, &status);
        z = oskar_mem_create(type, loc, num, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Set RA and Dec of point.
        oskar_mem_set_value_real(ra,  40 * D2R, 0, 0, &status);
        oskar_mem_set_value_real(dec, 65 * D2R, 0, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute l, m, n directions.
        oskar_convert_apparent_ra_dec_to_relative_direction_cosines(num,
                ra, dec, ra0, dec0, l, m, n, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute ENU directions.
        oskar_convert_relative_direction_cosines_to_enu_direction_cosines(
                x, y, z, num, l, m, n, ha0, dec0, lat, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check values.
        ASSERT_NEAR(oskar_mem_double(x, &status)[0],
                0.198407255801263, tol);
        ASSERT_NEAR(oskar_mem_double(y, &status)[0],
                -0.0918661536386987, tol);
        ASSERT_NEAR(oskar_mem_double(z, &status)[0],
                0.97580488349928, tol);

        // Free input data.
        oskar_mem_free(ra, &status);
        oskar_mem_free(dec, &status);
        oskar_mem_free(l, &status);
        oskar_mem_free(m, &status);
        oskar_mem_free(n, &status);
        free(ra); // FIXME Remove after updating oskar_mem_free().
        free(dec); // FIXME Remove after updating oskar_mem_free().
        free(l); // FIXME Remove after updating oskar_mem_free().
        free(m); // FIXME Remove after updating oskar_mem_free().
        free(n); // FIXME Remove after updating oskar_mem_free().

        // GPU version.
        loc = OSKAR_LOCATION_GPU;

        ra = oskar_mem_create(type, loc, num, &status);
        dec = oskar_mem_create(type, loc, num, &status);
        l = oskar_mem_create(type, loc, num, &status);
        m = oskar_mem_create(type, loc, num, &status);
        n = oskar_mem_create(type, loc, num, &status);
        x_gpu = oskar_mem_create(type, loc, num, &status);
        y_gpu = oskar_mem_create(type, loc, num, &status);
        z_gpu = oskar_mem_create(type, loc, num, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Set RA and Dec of point.
        oskar_mem_set_value_real(ra,  40 * D2R, 0, 0, &status);
        oskar_mem_set_value_real(dec, 65 * D2R, 0, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute l, m, n directions.
        oskar_convert_apparent_ra_dec_to_relative_direction_cosines(num,
                ra, dec, ra0, dec0, l, m, n, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute ENU directions.
        oskar_convert_relative_direction_cosines_to_enu_direction_cosines(
                x_gpu, y_gpu, z_gpu, num, l, m, n, ha0, dec0, lat, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check values.
        double max_err, mean_err;
        oskar_mem_evaluate_relative_error(x_gpu, x,
                0, &max_err, &mean_err, 0, &status);
        ASSERT_LT(max_err, tol);
        ASSERT_LT(mean_err, tol);
        oskar_mem_evaluate_relative_error(y_gpu, y,
                0, &max_err, &mean_err, 0, &status);
        ASSERT_LT(max_err, tol);
        ASSERT_LT(mean_err, tol);
        oskar_mem_evaluate_relative_error(z_gpu, z,
                0, &max_err, &mean_err, 0, &status);
        ASSERT_LT(max_err, tol);
        ASSERT_LT(mean_err, tol);

        // Free input data.
        oskar_mem_free(ra, &status);
        oskar_mem_free(dec, &status);
        oskar_mem_free(l, &status);
        oskar_mem_free(m, &status);
        oskar_mem_free(n, &status);
        free(ra); // FIXME Remove after updating oskar_mem_free().
        free(dec); // FIXME Remove after updating oskar_mem_free().
        free(l); // FIXME Remove after updating oskar_mem_free().
        free(m); // FIXME Remove after updating oskar_mem_free().
        free(n); // FIXME Remove after updating oskar_mem_free().

        // Free xyz directions.
        oskar_mem_free(x, &status);
        oskar_mem_free(y, &status);
        oskar_mem_free(z, &status);
        oskar_mem_free(x_gpu, &status);
        oskar_mem_free(y_gpu, &status);
        oskar_mem_free(z_gpu, &status);
        free(x); // FIXME Remove after updating oskar_mem_free().
        free(y); // FIXME Remove after updating oskar_mem_free().
        free(z); // FIXME Remove after updating oskar_mem_free().
        free(x_gpu); // FIXME Remove after updating oskar_mem_free().
        free(y_gpu); // FIXME Remove after updating oskar_mem_free().
        free(z_gpu); // FIXME Remove after updating oskar_mem_free().
    }
}
