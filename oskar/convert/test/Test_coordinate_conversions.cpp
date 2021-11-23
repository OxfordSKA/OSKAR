/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "convert/oskar_convert_cirs_relative_directions_to_enu_directions.h"
#include "convert/oskar_convert_lon_lat_to_relative_directions.h"
#include "convert/oskar_convert_lon_lat_to_xyz.h"
#include "convert/oskar_convert_relative_directions_to_lon_lat.h"
#include "convert/oskar_convert_xyz_to_lon_lat.h"
#include "math/oskar_cmath.h"
#include "math/oskar_evaluate_image_lm_grid.h"
#include "math/oskar_linspace.h"
#include "utility/oskar_get_error_string.h"

#include <cstdlib>
#include <cstdio>
#include <vector>

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

    oskar_convert_lon_lat_to_xyz_d(num_pts,
            &lon_in[0], &lat_in[0], &x[0], &y[0], &z[0]);
    oskar_convert_xyz_to_lon_lat_d(num_pts,
            &x[0], &y[0], &z[0], &lon_out[0], &lat_out[0]);

    ASSERT_NEAR(lon_in[0], lon_out[0], delta);
    ASSERT_NEAR(lat_in[0], lat_out[0], delta);
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
    std::vector<double> l_1(num_points), m_1(num_points);
    std::vector<double> l_2(num_points), m_2(num_points), n_2(num_points);
    std::vector<double> ra(num_points), dec(num_points);
    oskar_evaluate_image_lm_grid_d(num_l, num_m, fov_lon_deg * M_PI / 180.0,
            fov_lat_deg * M_PI / 180.0, &l_1[0], &m_1[0]);

    // Convert from l,m grid to spherical coordinates.
    const double cos_dec0 = cos(dec0);
    const double sin_dec0 = sin(dec0);
    oskar_convert_relative_directions_to_lon_lat_2d_d(num_points,
            &l_1[0], &m_1[0], 0, ra0, cos_dec0, sin_dec0, &ra[0], &dec[0]);

    // Check reverse direction.
    oskar_convert_lon_lat_to_relative_directions_3d_d(num_points, &ra[0],
            &dec[0], ra0, cos(dec0), sin(dec0), &l_2[0], &m_2[0], &n_2[0]);

    for (int i = 0; i < num_points; ++i)
    {
        ASSERT_NEAR(l_1[i], l_2[i], 1e-15);
        ASSERT_NEAR(m_1[i], m_2[i], 1e-15);
        const double n_1 = sqrt(1.0 - l_1[i]*l_1[i] - m_1[i]*m_1[i]);
        ASSERT_NEAR(n_1, n_2[i], 1e-15);
    }
}


TEST(coordinate_conversions, cirs_relative_directions_to_enu_directions)
{
    int type = 0, loc = 0, num = 1, status = 0;
    double tol = 0.0;

    // Observer's location and ERA.
    const double lon = 0.0;
    const double lat = 73 * D2R;
    const double era = 12 * D2R;

    // RA and Dec of phase centre.
    const double ra0  = 36 * D2R;
    const double dec0 = 60 * D2R;

    // Single precision.
    {
        oskar_Mem *ra = 0, *dec = 0, *l = 0, *m = 0, *n = 0;
        oskar_Mem *x = 0, *y = 0, *z = 0, *x_gpu = 0, *y_gpu = 0, *z_gpu = 0;
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
        oskar_mem_set_value_real(ra,  40 * D2R, 0, num, &status);
        oskar_mem_set_value_real(dec, 65 * D2R, 0, num, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute l, m, n directions.
        oskar_convert_lon_lat_to_relative_directions(num,
                ra, dec, ra0, dec0, l, m, n, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute ENU directions.
        oskar_convert_cirs_relative_directions_to_enu_directions(0, 0, 0,
                num, l, m, n, ra0, dec0, lon, lat, era, 0.0, 0.0, 0.0,
                0, x, y, z, &status);
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

        // Device/GPU version.
#ifdef OSKAR_HAVE_CUDA
        loc = OSKAR_GPU;
#endif

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
        oskar_mem_set_value_real(ra,  40 * D2R, 0, num, &status);
        oskar_mem_set_value_real(dec, 65 * D2R, 0, num, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute l, m, n directions.
        oskar_convert_lon_lat_to_relative_directions(num,
                ra, dec, ra0, dec0, l, m, n, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute ENU directions.
        oskar_convert_cirs_relative_directions_to_enu_directions(0, 0, 0,
                num, l, m, n, ra0, dec0, lon, lat, era, 0.0, 0.0, 0.0,
                0, x_gpu, y_gpu, z_gpu, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check values.
        double max_err = 0.0, mean_err = 0.0;
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
        oskar_Mem *ra = 0, *dec = 0, *l = 0, *m = 0, *n = 0;
        oskar_Mem *x = 0, *y = 0, *z = 0, *x_gpu = 0, *y_gpu = 0, *z_gpu = 0;
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
        oskar_mem_set_value_real(ra,  40 * D2R, 0, num, &status);
        oskar_mem_set_value_real(dec, 65 * D2R, 0, num, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute l, m, n directions.
        oskar_convert_lon_lat_to_relative_directions(num,
                ra, dec, ra0, dec0, l, m, n, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute ENU directions.
        oskar_convert_cirs_relative_directions_to_enu_directions(0, 0, 0,
                num, l, m, n, ra0, dec0, lon, lat, era, 0.0, 0.0, 0.0,
                0, x, y, z, &status);
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

        // Device/GPU version.
#ifdef OSKAR_HAVE_CUDA
        loc = OSKAR_GPU;
#endif

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
        oskar_mem_set_value_real(ra,  40 * D2R, 0, num, &status);
        oskar_mem_set_value_real(dec, 65 * D2R, 0, num, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute l, m, n directions.
        oskar_convert_lon_lat_to_relative_directions(num,
                ra, dec, ra0, dec0, l, m, n, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Compute ENU directions.
        oskar_convert_cirs_relative_directions_to_enu_directions(0, 0, 0,
                num, l, m, n, ra0, dec0, lon, lat, era, 0.0, 0.0, 0.0,
                0, x_gpu, y_gpu, z_gpu, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check values.
        double max_err = 0.0, mean_err = 0.0;
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
