/*
 * Copyright (c) 2012-2014, The University of Oxford
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
#include <oskar_convert_cirs_relative_directions_to_enu_directions.h>
#include <oskar_convert_lon_lat_to_relative_directions.h>
#include <oskar_convert_lon_lat_to_xyz.h>
#include <oskar_convert_relative_directions_to_lon_lat.h>
#include <oskar_convert_xyz_to_lon_lat.h>
#include <oskar_evaluate_image_lm_grid.h>
#include <oskar_get_error_string.h>

#include <oskar_cmath.h>
#include <cstdlib>
#include <cstdio>

#define D2R M_PI/180.0

TEST(coordinate_conversions, lon_lat_to_xyz)
{
    int num_pts = 1;
    double *x, *y, *z, *lon_in, *lat_in, *lon_out, *lat_out;
    x = (double*)malloc(num_pts * sizeof(double));
    y = (double*)malloc(num_pts * sizeof(double));
    z = (double*)malloc(num_pts * sizeof(double));
    lon_in = (double*)malloc(num_pts * sizeof(double));
    lat_in = (double*)malloc(num_pts * sizeof(double));
    lon_out = (double*)malloc(num_pts * sizeof(double));
    lat_out = (double*)malloc(num_pts * sizeof(double));
    double delta = 1e-8;

    lon_in[0] = 50.0 * M_PI/180.0;
    lat_in[0] = 30.0 * M_PI/180.0;

    oskar_convert_lon_lat_to_xyz_d(num_pts, lon_in, lat_in, x, y, z);
    oskar_convert_xyz_to_lon_lat_d(num_pts, x, y, z, lon_out, lat_out);

    ASSERT_NEAR(lon_in[0], lon_out[0], delta);
    ASSERT_NEAR(lat_in[0], lat_out[0], delta);
    free(x);
    free(y);
    free(z);
    free(lon_in);
    free(lat_in);
    free(lon_out);
    free(lat_out);
}

TEST(coordinate_conversions, ra_dec_to_directions)
{
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
    double *l_1, *m_1, *l_2, *m_2, *n_2, *ra, *dec;
    l_1 = (double*)malloc(num_points * sizeof(double));
    m_1 = (double*)malloc(num_points * sizeof(double));
    l_2 = (double*)malloc(num_points * sizeof(double));
    m_2 = (double*)malloc(num_points * sizeof(double));
    n_2 = (double*)malloc(num_points * sizeof(double));
    ra  = (double*)malloc(num_points * sizeof(double));
    dec = (double*)malloc(num_points * sizeof(double));
    oskar_evaluate_image_lm_grid_d(num_l, num_m, fov_lon_deg * M_PI / 180.0,
            fov_lat_deg * M_PI / 180.0, l_1, m_1);

    // Convert from l,m grid to spherical coordinates.
    oskar_convert_relative_directions_to_lon_lat_2d_d(num_points,
            l_1, m_1, ra0, dec0, ra, dec);

    // Check reverse direction.
    oskar_convert_lon_lat_to_relative_directions_d(num_points,
            ra, dec, ra0, dec0, l_2, m_2, n_2);

    for (int i = 0; i < num_points; ++i)
    {
        ASSERT_NEAR(l_1[i], l_2[i], 1e-15);
        ASSERT_NEAR(m_1[i], m_2[i], 1e-15);
        double n_1 = sqrt(1.0 - l_1[i]*l_1[i] - m_1[i]*m_1[i]);
        ASSERT_NEAR(n_1, n_2[i], 1e-15);
    }
    free(l_1);
    free(m_1);
    free(l_2);
    free(m_2);
    free(n_2);
    free(ra);
    free(dec);
}


TEST(coordinate_conversions, cirs_relative_directions_to_enu_directions)
{
    int type, loc, num = 1, status = 0;
    double lon, lat, era, ra0, dec0, tol;

    // Observer's location and ERA.
    lon = 0.0;
    lat = 73 * D2R;
    era = 12 * D2R;

    // RA and Dec of phase centre.
    ra0  = 36 * D2R;
    dec0 = 60 * D2R;

    // Single precision.
    {
        oskar_Mem *ra, *dec, *l, *m, *n, *x, *y, *z, *x_gpu, *y_gpu, *z_gpu;
        type = OSKAR_SINGLE;
        loc = OSKAR_CPU;
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
        oskar_convert_lon_lat_to_relative_directions(num,
                ra, dec, ra0, dec0, l, m, n, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute ENU directions.
        oskar_convert_cirs_relative_directions_to_enu_directions(num, l, m, n,
                ra0, dec0, lon, lat, era, 0.0, 0.0, 0.0, x, y, z, &status);
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

        // GPU version.
        loc = OSKAR_GPU;

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
        oskar_convert_lon_lat_to_relative_directions(num,
                ra, dec, ra0, dec0, l, m, n, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute ENU directions.
        oskar_convert_cirs_relative_directions_to_enu_directions(num, l, m, n,
                ra0, dec0, lon, lat, era, 0.0, 0.0, 0.0, x_gpu, y_gpu, z_gpu,
                &status);
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

        // Free xyz directions.
        oskar_mem_free(x, &status);
        oskar_mem_free(y, &status);
        oskar_mem_free(z, &status);
        oskar_mem_free(x_gpu, &status);
        oskar_mem_free(y_gpu, &status);
        oskar_mem_free(z_gpu, &status);
    }

    // Double precision.
    {
        oskar_Mem *ra, *dec, *l, *m, *n, *x, *y, *z, *x_gpu, *y_gpu, *z_gpu;
        type = OSKAR_DOUBLE;
        loc = OSKAR_CPU;
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
        oskar_convert_lon_lat_to_relative_directions(num,
                ra, dec, ra0, dec0, l, m, n, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute ENU directions.
        oskar_convert_cirs_relative_directions_to_enu_directions(num, l, m, n,
                ra0, dec0, lon, lat, era, 0.0, 0.0, 0.0, x, y, z, &status);
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

        // GPU version.
        loc = OSKAR_GPU;

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
        oskar_convert_lon_lat_to_relative_directions(num,
                ra, dec, ra0, dec0, l, m, n, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute ENU directions.
        oskar_convert_cirs_relative_directions_to_enu_directions(num, l, m, n,
                ra0, dec0, lon, lat, era, 0.0, 0.0, 0.0, x_gpu, y_gpu, z_gpu,
                &status);
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

        // Free xyz directions.
        oskar_mem_free(x, &status);
        oskar_mem_free(y, &status);
        oskar_mem_free(z, &status);
        oskar_mem_free(x_gpu, &status);
        oskar_mem_free(y_gpu, &status);
        oskar_mem_free(z_gpu, &status);
    }
}
