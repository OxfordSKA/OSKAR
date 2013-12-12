/*
 * Copyright (c) 2013, The University of Oxford
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

#include <cuda_runtime_api.h>

#include <oskar_station.h>
#include <oskar_evaluate_array_pattern.h>
#include <oskar_evaluate_array_pattern_hierarchical.h>
#include <oskar_evaluate_beam_horizon_direction.h>
#include <oskar_convert_apparent_ra_dec_to_relative_direction_cosines.h>
#include <oskar_convert_relative_direction_cosines_to_enu_direction_cosines.h>
#include <oskar_evaluate_element_weights_dft.h>
#include <oskar_evaluate_image_lon_lat_grid.h>
#include <oskar_image_free.h>
#include <oskar_image_init.h>
#include <oskar_image_resize.h>
#include <oskar_get_error_string.h>

#define TIMER_ENABLE 1
#include "utility/timer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace std;

static void check_images(const oskar_Image* image1, const oskar_Image* image2)
{
    int status = 0;

    if (image1->width != image2->width || image1->height != image2->height)
    {
        FAIL() << "Inconsistent image dimensions.";
        return;
    }
    if (image1->num_pols != image2->num_pols)
    {
        FAIL() << "Inconsistent polarisation dimensions.";
        return;
    }
    if (image1->num_times != image2->num_times)
    {
        FAIL() << "Inconsistent time dimensions.";
        return;
    }
    if (image1->num_channels != image2->num_channels)
    {
        FAIL() << "Inconsistent frequency dimensions.";
        return;
    }
    if ((oskar_mem_type(&image1->data) & OSKAR_COMPLEX) !=
            (oskar_mem_type(&image2->data) & OSKAR_COMPLEX))
    {
        FAIL() << "Inconsistent data types (complex flag).";
        return;
    }
    if ((oskar_mem_type(&image1->data) & OSKAR_MATRIX) !=
            (oskar_mem_type(&image2->data) & OSKAR_MATRIX))
    {
        FAIL() << "Inconsistent data types (matrix flag).";
        return;
    }

    /* Check image contents are the same, to appropriate precision. */
    double min_rel_error = 0., max_rel_error = 0.;
    double avg_rel_error = 0., std_rel_error = 0.;
    oskar_mem_evaluate_relative_error(&image1->data, &image2->data,
            &min_rel_error, &max_rel_error, &avg_rel_error, &std_rel_error,
            &status);
    ASSERT_EQ(0, status);
    EXPECT_LT(max_rel_error, 1e-5);
    EXPECT_LT(avg_rel_error, 1e-5);
}

static void set_up_beam_pattern(oskar_Image* bp, int type, bool polarised,
        int image_size, double fov_deg, double ra_deg, double dec_deg,
        double freq_hz, double mjd, int* status)
{
    int num_times, num_channels, num_pols;
    num_times    = 1;
    num_channels = 1;
    num_pols     = polarised ? 4 : 1;

    /* Initialise complex image cube. */
    oskar_image_init(bp, type | OSKAR_COMPLEX, OSKAR_LOCATION_CPU, status);
    oskar_image_resize(bp, image_size, image_size, num_pols, num_times,
            num_channels, status);

    /* Set beam pattern meta-data. */
    bp->image_type         = (num_pols == 1) ?
            OSKAR_IMAGE_TYPE_BEAM_SCALAR : OSKAR_IMAGE_TYPE_BEAM_POLARISED;
    bp->centre_ra_deg      = ra_deg;
    bp->centre_dec_deg     = dec_deg;
    bp->fov_ra_deg         = fov_deg;
    bp->fov_dec_deg        = fov_deg;
    bp->freq_start_hz      = freq_hz;
    bp->freq_inc_hz        = 1;
    bp->time_inc_sec       = 1;
    bp->time_start_mjd_utc = mjd;
}

static oskar_Station* set_up_station1(int num_x, int num_y,
        int type, double beam_ra_deg, double beam_dec_deg, int* status)
{
    oskar_Station* station;

    /* Generator parameters. */
    double sep_m = 1.0;
    int dummy = 0, ix, iy, i;

    /* Initialise the station model. */
    station = oskar_station_create(type, OSKAR_LOCATION_CPU,
            num_x * num_y, status);

    /* Generate a square station. */
    for (iy = 0, i = 0; iy < num_y; ++iy)
    {
        for (ix = 0; ix < num_x; ++ix, ++i)
        {
            float x, y;
            x = ix * sep_m - (num_x - 1) * sep_m / 2;
            y = iy * sep_m - (num_y - 1) * sep_m / 2;

            oskar_station_set_element_coords(station, i,
                    x, y, 0.0, 0.0, 0.0, 0.0, status);
            oskar_station_set_element_errors(station, i,
                    1.0, 0.0, 0.0, 0.0, status);
            oskar_station_set_element_weight(station, i,
                    1.0, 0.0, status);
            oskar_station_set_element_orientation(station, i,
                    90.0, 0.0, status);
        }
    }

    /* Load the station file. */
    oskar_station_analyse(station, &dummy, status);

    /* Set meta-data. */
    oskar_station_set_position(station, 0.0, 70.0 * M_PI / 180.0, 0.0);
    oskar_station_set_phase_centre(station, OSKAR_SPHERICAL_TYPE_EQUATORIAL,
            beam_ra_deg * M_PI / 180.0, beam_dec_deg * M_PI / 180.0);
    return station;
}

static void set_up_pointing(oskar_Mem** weights, oskar_Mem** x, oskar_Mem** y,
        oskar_Mem** z, const oskar_Station* station, const oskar_Mem* lon,
        const oskar_Mem* lat, double gast, double freq_hz, int* status)
{
    double beam_x, beam_y, beam_z, st_lat, last, wavenumber;
    int type, location, num_elements, num_points;
    oskar_Mem *l, *m, *n;

    type = oskar_station_precision(station);
    location = oskar_station_location(station);
    num_elements = oskar_station_num_elements(station);
    num_points = oskar_mem_length(lon);
    wavenumber = 2.0 * M_PI * freq_hz / 299792458.0;
    last = gast + oskar_station_longitude_rad(station);
    st_lat = oskar_station_latitude_rad(station);
    *weights = oskar_mem_create(type | OSKAR_COMPLEX, location, num_elements,
            status);
    *x = oskar_mem_create(type, location, num_points, status);
    *y = oskar_mem_create(type, location, num_points, status);
    *z = oskar_mem_create(type, location, num_points, status);
    l = oskar_mem_create(type, location, num_points, status);
    m = oskar_mem_create(type, location, num_points, status);
    n = oskar_mem_create(type, location, num_points, status);
    oskar_evaluate_beam_horizon_direction(&beam_x, &beam_y, &beam_z, station,
            gast, status);
    oskar_convert_apparent_ra_dec_to_relative_direction_cosines(num_points,
            lon, lat, 0.0, 0.0, l, m, n, status);
    oskar_convert_relative_direction_cosines_to_enu_direction_cosines(
            *x, *y, *z, num_points, l, m, n, last, 0.0, st_lat, status);
    oskar_evaluate_element_weights_dft(*weights, num_elements, wavenumber,
            oskar_station_element_x_weights_const(station),
            oskar_station_element_y_weights_const(station),
            oskar_station_element_z_weights_const(station),
            beam_x, beam_y, beam_z, status);
    oskar_mem_free(l, status);
    oskar_mem_free(m, status);
    oskar_mem_free(n, status);
    free(l); // FIXME Remove after updating oskar_mem_free().
    free(m); // FIXME Remove after updating oskar_mem_free().
    free(n); // FIXME Remove after updating oskar_mem_free().
}

static void run_array_pattern(oskar_Image* bp,
        const oskar_Station* station, const oskar_Mem* lon,
        const oskar_Mem* lat, double gast, double freq_hz,
        const char* message, int* status)
{
    oskar_Mem *w, *x, *y, *z, *pattern;
    int num_pixels, location;
    double wavenumber;

    /* Get the meta-data. */
    num_pixels = (int)oskar_mem_length(lon);
    location = oskar_station_location(station);
    wavenumber = 2.0 * M_PI * freq_hz / 299792458.0;

    /* Initialise temporary arrays. */
    pattern = oskar_mem_create(oskar_mem_type(&bp->data), location, num_pixels,
            status);
    ASSERT_EQ(0, *status) << oskar_get_error_string(*status);
    set_up_pointing(&w, &x, &y, &z, station, lon, lat, gast, freq_hz, status);
    ASSERT_EQ(0, *status) << oskar_get_error_string(*status);
    TIMER_START
    oskar_evaluate_array_pattern(pattern, wavenumber, station, num_pixels,
            x, y, z, w, status);
    cudaDeviceSynchronize();
    TIMER_STOP("%s", message)
    ASSERT_EQ(0, *status) << oskar_get_error_string(*status);
    oskar_mem_free(w, status);
    oskar_mem_free(x, status);
    oskar_mem_free(y, status);
    oskar_mem_free(z, status);
    free(w); // FIXME Remove after updating oskar_mem_free().
    free(x); // FIXME Remove after updating oskar_mem_free().
    free(y); // FIXME Remove after updating oskar_mem_free().
    free(z); // FIXME Remove after updating oskar_mem_free().

    oskar_mem_insert(&bp->data, pattern, 0, status);
    ASSERT_EQ(0, *status) << oskar_get_error_string(*status);
    oskar_mem_free(pattern, status);
    free(pattern); // FIXME Remove after updating oskar_mem_free().
}

static void run_array_pattern_hierarchical(oskar_Image* bp,
        const oskar_Station* station, const oskar_Mem* lon,
        const oskar_Mem* lat, double gast, double freq_hz,
        const char* message, int* status)
{
    oskar_Mem *w, *x, *y, *z, *ones, *pattern;
    int num_pixels, location;
    double wavenumber;

    /* Get the meta-data. */
    num_pixels = (int)oskar_mem_length(lon);
    location = oskar_station_location(station);
    wavenumber = 2.0 * M_PI * freq_hz / 299792458.0;

    /* Initialise temporary array. */
    pattern = oskar_mem_create(oskar_mem_type(&bp->data), location,
            num_pixels, status);

    /* Create a fake complex "signal" vector of ones. */
    ones = oskar_mem_create(oskar_mem_type(&bp->data), location,
            num_pixels * oskar_station_num_elements(station), status);
    oskar_mem_set_value_real(ones, 1.0, 0, 0, status);
    set_up_pointing(&w, &x, &y, &z, station, lon, lat, gast, freq_hz, status);
    TIMER_START
    oskar_evaluate_array_pattern_hierarchical(pattern, wavenumber, station,
            num_pixels, x, y, z, ones, w, status);
    cudaDeviceSynchronize();
    TIMER_STOP("%s", message)
    oskar_mem_free(w, status);
    oskar_mem_free(x, status);
    oskar_mem_free(y, status);
    oskar_mem_free(z, status);
    oskar_mem_free(ones, status);
    free(w); // FIXME Remove after updating oskar_mem_free().
    free(x); // FIXME Remove after updating oskar_mem_free().
    free(y); // FIXME Remove after updating oskar_mem_free().
    free(z); // FIXME Remove after updating oskar_mem_free().
    free(ones); // FIXME Remove after updating oskar_mem_free().

    if (oskar_mem_is_scalar(pattern))
    {
        oskar_mem_insert(&bp->data, pattern, 0, status);
    }
    else
    {
        /* Copy beam pattern for re-ordering. */
        oskar_Mem *pattern_temp = oskar_mem_create_copy(pattern,
                OSKAR_LOCATION_CPU, status);
        ASSERT_EQ(0, *status) << oskar_get_error_string(*status);

        /* Re-order the polarisation data. */
        if (oskar_mem_precision(pattern) == OSKAR_SINGLE)
        {
            float2* p = oskar_mem_float2(&bp->data, status);
            float4c* tc = oskar_mem_float4c(pattern_temp, status);
            for (int i = 0; i < num_pixels; ++i)
            {
                p[i]                  = tc[i].a; // theta_X
                p[i +     num_pixels] = tc[i].b; // phi_X
                p[i + 2 * num_pixels] = tc[i].c; // theta_Y
                p[i + 3 * num_pixels] = tc[i].d; // phi_Y
            }
        }
        else if (oskar_mem_precision(pattern) == OSKAR_DOUBLE)
        {
            double2* p = oskar_mem_double2(&bp->data, status);
            double4c* tc = oskar_mem_double4c(pattern_temp, status);
            for (int i = 0; i < num_pixels; ++i)
            {
                p[i]                  = tc[i].a; // theta_X
                p[i +     num_pixels] = tc[i].b; // phi_X
                p[i + 2 * num_pixels] = tc[i].c; // theta_Y
                p[i + 3 * num_pixels] = tc[i].d; // phi_Y
            }
        }
        oskar_mem_free(pattern_temp, status);
        free(pattern_temp); // FIXME Remove after updating oskar_mem_free().
    }

    oskar_mem_free(pattern, status);
    free(pattern); // FIXME Remove after updating oskar_mem_free().
}

TEST(evaluate_array_pattern, test)
{
    /* Inputs. */
    int station_side = 8;
    int image_side = 128;
    double ra_deg = 0.0;
    double dec_deg = 80.0;
    double fov_deg = 10.0;
    double freq_hz = 100e6;
    double gast = 0.0;
    double mjd = 0.0;

    bool polarised;
    int status = 0, type = 0;
    oskar_Station *station_cpu_f, *station_cpu_d;
    oskar_Station *station_gpu_f, *station_gpu_d;
    oskar_Mem *lon_cpu_f, *lat_cpu_f, *lon_cpu_d, *lat_cpu_d;
    oskar_Mem *lon_gpu_f, *lat_gpu_f, *lon_gpu_d, *lat_gpu_d;
    oskar_Image bp_o2c_2d_cpu_f, bp_o2c_2d_cpu_d;
    oskar_Image bp_o2c_2d_gpu_f, bp_o2c_2d_gpu_d;
    oskar_Image bp_o2c_3d_cpu_f, bp_o2c_3d_cpu_d;
    oskar_Image bp_o2c_3d_gpu_f, bp_o2c_3d_gpu_d;
    oskar_Image bp_c2c_2d_cpu_f, bp_c2c_2d_cpu_d;
    oskar_Image bp_c2c_2d_gpu_f, bp_c2c_2d_gpu_d;
    oskar_Image bp_c2c_3d_cpu_f, bp_c2c_3d_cpu_d;
    oskar_Image bp_c2c_3d_gpu_f, bp_c2c_3d_gpu_d;
    oskar_Image bp_m2m_2d_cpu_f, bp_m2m_2d_cpu_d;
    oskar_Image bp_m2m_2d_gpu_f, bp_m2m_2d_gpu_d;
    oskar_Image bp_m2m_3d_cpu_f, bp_m2m_3d_cpu_d;
    oskar_Image bp_m2m_3d_gpu_f, bp_m2m_3d_gpu_d;

    /* Convert inputs. */
    double ra_rad  = ra_deg  * M_PI / 180.0;
    double dec_rad = dec_deg * M_PI / 180.0;
    double fov_rad = fov_deg * M_PI / 180.0;
    int num_pixels = image_side * image_side;

    /* Set up station models. */
    station_cpu_f = set_up_station1(station_side, station_side, OSKAR_SINGLE,
            ra_deg, dec_deg, &status);
    station_cpu_d = set_up_station1(station_side, station_side, OSKAR_DOUBLE,
            ra_deg, dec_deg, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    station_gpu_f = oskar_station_create_copy(station_cpu_f,
            OSKAR_LOCATION_GPU, &status);
    station_gpu_d = oskar_station_create_copy(station_cpu_d,
            OSKAR_LOCATION_GPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    /* Set up longitude/latitude grids. */
    type = OSKAR_SINGLE;
    lon_cpu_f = oskar_mem_create(type, OSKAR_LOCATION_CPU, num_pixels, &status);
    lat_cpu_f = oskar_mem_create(type, OSKAR_LOCATION_CPU, num_pixels, &status);
    oskar_evaluate_image_lon_lat_grid(lon_cpu_f, lat_cpu_f, image_side,
            image_side, fov_rad, fov_rad, ra_rad, dec_rad, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    type = OSKAR_DOUBLE;
    lon_cpu_d = oskar_mem_create(type, OSKAR_LOCATION_CPU, num_pixels, &status);
    lat_cpu_d = oskar_mem_create(type, OSKAR_LOCATION_CPU, num_pixels, &status);
    oskar_evaluate_image_lon_lat_grid(lon_cpu_d, lat_cpu_d, image_side,
            image_side, fov_rad, fov_rad, ra_rad, dec_rad, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    lon_gpu_f = oskar_mem_create_copy(lon_cpu_f, OSKAR_LOCATION_GPU, &status);
    lon_gpu_d = oskar_mem_create_copy(lon_cpu_d, OSKAR_LOCATION_GPU, &status);
    lat_gpu_f = oskar_mem_create_copy(lat_cpu_f, OSKAR_LOCATION_GPU, &status);
    lat_gpu_d = oskar_mem_create_copy(lat_cpu_d, OSKAR_LOCATION_GPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    /* Set up beam patterns. */
    type = OSKAR_SINGLE_COMPLEX;
    polarised = false;
    set_up_beam_pattern(&bp_o2c_2d_cpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_o2c_2d_gpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_o2c_3d_cpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_o2c_3d_gpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_c2c_2d_cpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_c2c_2d_gpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_c2c_3d_cpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_c2c_3d_gpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    polarised = true;
    type = OSKAR_SINGLE_COMPLEX_MATRIX;
    set_up_beam_pattern(&bp_m2m_2d_cpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_m2m_2d_gpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_m2m_3d_cpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_m2m_3d_gpu_f, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    type = OSKAR_DOUBLE_COMPLEX;
    polarised = false;
    set_up_beam_pattern(&bp_o2c_2d_cpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_o2c_2d_gpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_o2c_3d_cpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_o2c_3d_gpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_c2c_2d_cpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_c2c_2d_gpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_c2c_3d_cpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_c2c_3d_gpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    polarised = true;
    type = OSKAR_DOUBLE_COMPLEX_MATRIX;
    set_up_beam_pattern(&bp_m2m_2d_cpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_m2m_2d_gpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_m2m_3d_cpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    set_up_beam_pattern(&bp_m2m_3d_gpu_d, type, polarised, image_side, fov_deg,
            ra_deg, dec_deg, freq_hz, mjd, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    /* Run the tests... */
    ASSERT_EQ(0, oskar_station_array_is_3d(station_cpu_f));
    ASSERT_EQ(0, oskar_station_array_is_3d(station_gpu_f));
    run_array_pattern(&bp_o2c_2d_cpu_f, station_cpu_f, lon_cpu_f, lat_cpu_f,
            gast, freq_hz, "Single, o2c, CPU, 2D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern(&bp_o2c_2d_gpu_f, station_gpu_f, lon_gpu_f, lat_gpu_f,
            gast, freq_hz, "Single, o2c, GPU, 2D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern(&bp_o2c_2d_cpu_d, station_cpu_d, lon_cpu_d, lat_cpu_d,
            gast, freq_hz, "Double, o2c, CPU, 2D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern(&bp_o2c_2d_gpu_d, station_gpu_d, lon_gpu_d, lat_gpu_d,
            gast, freq_hz, "Double, o2c, GPU, 2D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern_hierarchical(&bp_c2c_2d_cpu_f, station_cpu_f,
            lon_cpu_f, lat_cpu_f, gast, freq_hz, "Single, c2c, CPU, 2D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern_hierarchical(&bp_c2c_2d_gpu_f, station_gpu_f,
            lon_gpu_f, lat_gpu_f, gast, freq_hz, "Single, c2c, GPU, 2D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern_hierarchical(&bp_c2c_2d_cpu_d, station_cpu_d,
            lon_cpu_d, lat_cpu_d, gast, freq_hz, "Double, c2c, CPU, 2D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern_hierarchical(&bp_c2c_2d_gpu_d, station_gpu_d,
            lon_gpu_d, lat_gpu_d, gast, freq_hz, "Double, c2c, GPU, 2D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern_hierarchical(&bp_m2m_2d_cpu_f, station_cpu_f,
            lon_cpu_f, lat_cpu_f, gast, freq_hz, "Single, m2m, CPU, 2D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern_hierarchical(&bp_m2m_2d_gpu_f, station_gpu_f,
            lon_gpu_f, lat_gpu_f, gast, freq_hz, "Single, m2m, GPU, 2D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern_hierarchical(&bp_m2m_2d_cpu_d, station_cpu_d,
            lon_cpu_d, lat_cpu_d, gast, freq_hz, "Double, m2m, CPU, 2D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern_hierarchical(&bp_m2m_2d_gpu_d, station_gpu_d,
            lon_gpu_d, lat_gpu_d, gast, freq_hz, "Double, m2m, GPU, 2D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    /* Set 3D arrays. */
    oskar_station_set_element_coords(station_cpu_f, 0,
            0., 0., 1., 0., 0., 0., &status);
    oskar_station_set_element_coords(station_gpu_f, 0,
            0., 0., 1., 0., 0., 0., &status);
    oskar_station_set_element_coords(station_cpu_f, 0,
            0., 0., 0., 0., 0., 0., &status);
    oskar_station_set_element_coords(station_gpu_f, 0,
            0., 0., 0., 0., 0., 0., &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(1, oskar_station_array_is_3d(station_cpu_f));
    ASSERT_EQ(1, oskar_station_array_is_3d(station_gpu_f));
    run_array_pattern(&bp_o2c_2d_cpu_f, station_cpu_f, lon_cpu_f, lat_cpu_f,
            gast, freq_hz, "Single, o2c, CPU, 3D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern(&bp_o2c_2d_gpu_f, station_gpu_f, lon_gpu_f, lat_gpu_f,
            gast, freq_hz, "Single, o2c, GPU, 3D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern(&bp_o2c_2d_cpu_d, station_cpu_d, lon_cpu_d, lat_cpu_d,
            gast, freq_hz, "Double, o2c, CPU, 3D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern(&bp_o2c_2d_gpu_d, station_gpu_d, lon_gpu_d, lat_gpu_d,
            gast, freq_hz, "Double, o2c, GPU, 3D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern_hierarchical(&bp_c2c_2d_cpu_f, station_cpu_f,
            lon_cpu_f, lat_cpu_f, gast, freq_hz, "Single, c2c, CPU, 3D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern_hierarchical(&bp_c2c_2d_gpu_f, station_gpu_f,
            lon_gpu_f, lat_gpu_f, gast, freq_hz, "Single, c2c, GPU, 3D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern_hierarchical(&bp_c2c_2d_cpu_d, station_cpu_d,
            lon_cpu_d, lat_cpu_d, gast, freq_hz, "Double, c2c, CPU, 3D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern_hierarchical(&bp_c2c_2d_gpu_d, station_gpu_d,
            lon_gpu_d, lat_gpu_d, gast, freq_hz, "Double, c2c, GPU, 3D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern_hierarchical(&bp_m2m_2d_cpu_f, station_cpu_f,
            lon_cpu_f, lat_cpu_f, gast, freq_hz, "Single, m2m, CPU, 3D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern_hierarchical(&bp_m2m_2d_gpu_f, station_gpu_f,
            lon_gpu_f, lat_gpu_f, gast, freq_hz, "Single, m2m, GPU, 3D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern_hierarchical(&bp_m2m_2d_cpu_d, station_cpu_d,
            lon_cpu_d, lat_cpu_d, gast, freq_hz, "Double, m2m, CPU, 3D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    run_array_pattern_hierarchical(&bp_m2m_2d_gpu_d, station_gpu_d,
            lon_gpu_d, lat_gpu_d, gast, freq_hz, "Double, m2m, GPU, 3D", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    /* Check for consistency. */
//    check_images(&bp_o2c_2d_cpu_f, &bp_o2c_2d_gpu_f);
//    check_images(&bp_o2c_3d_cpu_f, &bp_o2c_3d_gpu_f);
//    check_images(&bp_c2c_2d_cpu_f, &bp_c2c_2d_gpu_f);
//    check_images(&bp_c2c_3d_cpu_f, &bp_c2c_3d_gpu_f);
//    check_images(&bp_m2m_2d_cpu_f, &bp_m2m_2d_gpu_f);
//    check_images(&bp_m2m_3d_cpu_f, &bp_m2m_3d_gpu_f);
//    check_images(&bp_o2c_2d_cpu_d, &bp_o2c_2d_gpu_d);
//    check_images(&bp_o2c_3d_cpu_d, &bp_o2c_3d_gpu_d);
//    check_images(&bp_c2c_2d_cpu_d, &bp_c2c_2d_gpu_d);
//    check_images(&bp_c2c_3d_cpu_d, &bp_c2c_3d_gpu_d);
//    check_images(&bp_m2m_2d_cpu_d, &bp_m2m_2d_gpu_d);
//    check_images(&bp_m2m_3d_cpu_d, &bp_m2m_3d_gpu_d);

//    check_images(&bp_o2c_2d_cpu_d, &bp_o2c_2d_gpu_f);
//    check_images(&bp_o2c_3d_cpu_d, &bp_o2c_3d_gpu_f);
//    check_images(&bp_c2c_2d_cpu_d, &bp_c2c_2d_gpu_f);
//    check_images(&bp_c2c_3d_cpu_d, &bp_c2c_3d_gpu_f);
//    check_images(&bp_m2m_2d_cpu_d, &bp_m2m_2d_gpu_f);
//    check_images(&bp_m2m_3d_cpu_d, &bp_m2m_3d_gpu_f);
    check_images(&bp_o2c_2d_cpu_d, &bp_o2c_2d_gpu_d);
    check_images(&bp_o2c_3d_cpu_d, &bp_o2c_3d_gpu_d);
    check_images(&bp_c2c_2d_cpu_d, &bp_c2c_2d_gpu_d);
    check_images(&bp_c2c_3d_cpu_d, &bp_c2c_3d_gpu_d);
    check_images(&bp_m2m_2d_cpu_d, &bp_m2m_2d_gpu_d);
    check_images(&bp_m2m_3d_cpu_d, &bp_m2m_3d_gpu_d);


    /* Free images. */
    oskar_image_free(&bp_o2c_2d_cpu_f, &status);
    oskar_image_free(&bp_o2c_2d_gpu_f, &status);
    oskar_image_free(&bp_o2c_3d_cpu_f, &status);
    oskar_image_free(&bp_o2c_3d_gpu_f, &status);
    oskar_image_free(&bp_c2c_2d_cpu_f, &status);
    oskar_image_free(&bp_c2c_2d_gpu_f, &status);
    oskar_image_free(&bp_c2c_3d_cpu_f, &status);
    oskar_image_free(&bp_c2c_3d_gpu_f, &status);
    oskar_image_free(&bp_m2m_2d_cpu_f, &status);
    oskar_image_free(&bp_m2m_2d_gpu_f, &status);
    oskar_image_free(&bp_m2m_3d_cpu_f, &status);
    oskar_image_free(&bp_m2m_3d_gpu_f, &status);
    oskar_image_free(&bp_o2c_2d_cpu_d, &status);
    oskar_image_free(&bp_o2c_2d_gpu_d, &status);
    oskar_image_free(&bp_o2c_3d_cpu_d, &status);
    oskar_image_free(&bp_o2c_3d_gpu_d, &status);
    oskar_image_free(&bp_c2c_2d_cpu_d, &status);
    oskar_image_free(&bp_c2c_2d_gpu_d, &status);
    oskar_image_free(&bp_c2c_3d_cpu_d, &status);
    oskar_image_free(&bp_c2c_3d_gpu_d, &status);
    oskar_image_free(&bp_m2m_2d_cpu_d, &status);
    oskar_image_free(&bp_m2m_2d_gpu_d, &status);
    oskar_image_free(&bp_m2m_3d_cpu_d, &status);
    oskar_image_free(&bp_m2m_3d_gpu_d, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    /* Free longitude/latitude points. */
    oskar_mem_free(lon_cpu_f, &status);
    oskar_mem_free(lat_cpu_f, &status);
    oskar_mem_free(lon_gpu_f, &status);
    oskar_mem_free(lat_gpu_f, &status);
    oskar_mem_free(lon_cpu_d, &status);
    oskar_mem_free(lat_cpu_d, &status);
    oskar_mem_free(lon_gpu_d, &status);
    oskar_mem_free(lat_gpu_d, &status);
    free(lon_cpu_f); // FIXME Remove after updating oskar_mem_free().
    free(lat_cpu_f); // FIXME Remove after updating oskar_mem_free().
    free(lon_gpu_f); // FIXME Remove after updating oskar_mem_free().
    free(lat_gpu_f); // FIXME Remove after updating oskar_mem_free().
    free(lon_cpu_d); // FIXME Remove after updating oskar_mem_free().
    free(lat_cpu_d); // FIXME Remove after updating oskar_mem_free().
    free(lon_gpu_d); // FIXME Remove after updating oskar_mem_free().
    free(lat_gpu_d); // FIXME Remove after updating oskar_mem_free().

    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    /* Free station models. */
    oskar_station_free(station_gpu_f, &status);
    oskar_station_free(station_gpu_d, &status);
    oskar_station_free(station_cpu_f, &status);
    oskar_station_free(station_cpu_d, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}
