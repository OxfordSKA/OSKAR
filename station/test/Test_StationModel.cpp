/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include <oskar_station.h>
#include <oskar_get_error_string.h>

#include <cmath>
#include <cstdio>


TEST(Station, test_load_single)
{
    // Create the test file.
    int status = 0;
    const char* filename = "test.dat";
    FILE* file = fopen(filename, "w");
    int n_elements = 100;
    for (int i = 0; i < n_elements/2; ++i)
        fprintf(file, "%.6f %.6f %.6f\n", i/10.0f, i/12.0f, i/14.0f);
    fprintf(file, "\n"); // Add a blank line halfway.
    for (int i = n_elements/2; i < n_elements; ++i)
        fprintf(file, "%.6f,%.6f,%.6f\n", i/10.0f, i/12.0f, i/14.0f);
    fclose(file);

    // Load the data.
    oskar_Station* station = oskar_station_create(OSKAR_SINGLE,
            OSKAR_LOCATION_CPU, 0, &status);
    oskar_station_load_config(station, filename, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check the coordinates.
    float tol = 1e-6;
    ASSERT_EQ(n_elements, oskar_station_num_elements(station));
    for (int i = 0; i < n_elements; ++i)
    {
        EXPECT_NEAR(i/10.0f, oskar_mem_float(
                oskar_station_element_x_weights(station), &status)[i], tol);
        EXPECT_NEAR(i/12.0f, oskar_mem_float(
                oskar_station_element_y_weights(station), &status)[i], tol);
        EXPECT_NEAR(i/14.0f, oskar_mem_float(
                oskar_station_element_z_weights(station), &status)[i], tol);
        EXPECT_NEAR(1.0f, oskar_mem_float2(
                oskar_station_element_weight(station), &status)[i].x, tol);
        EXPECT_NEAR(0.0f, oskar_mem_float2(
                oskar_station_element_weight(station), &status)[i].y, tol);
        EXPECT_NEAR(1.0f, oskar_mem_float(
                oskar_station_element_gain(station), &status)[i], tol);
        EXPECT_NEAR(0.0f, oskar_mem_float(
                oskar_station_element_gain_error(station), &status)[i], tol);
        EXPECT_NEAR(0.0f, oskar_mem_float(
                oskar_station_element_phase_offset(station), &status)[i], tol);
        EXPECT_NEAR(0.0f, oskar_mem_float(
                oskar_station_element_phase_error(station), &status)[i], tol);
    }

    // Remove the test file.
    remove(filename);

    // Free memory.
    oskar_station_free(station, &status);
}

TEST(Station, test_load_double)
{
    // Create the test file.
    int status = 0;
    const char* filename = "test.dat";
    FILE* file = fopen(filename, "w");
    int n_elements = 100;
    for (int i = 0; i < n_elements/2; ++i)
        fprintf(file, "%.6f %.6f %.6f\n", i/10.0, i/12.0, i/14.0);
    fprintf(file, "\n"); // Add a blank line halfway.
    for (int i = n_elements/2; i < n_elements; ++i)
        fprintf(file, "%.6f,%.6f,%.6f\n", i/10.0, i/12.0, i/14.0);
    fclose(file);

    // Load the data.
    oskar_Station* station = oskar_station_create(OSKAR_DOUBLE,
            OSKAR_LOCATION_CPU, 0, &status);
    oskar_station_load_config(station, filename, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check the coordinates.
    double tol = 1e-6;
    ASSERT_EQ(n_elements, oskar_station_num_elements(station));
    for (int i = 0; i < n_elements; ++i)
    {
        EXPECT_NEAR(i/10.0, oskar_mem_double(
                oskar_station_element_x_weights(station), &status)[i], tol);
        EXPECT_NEAR(i/12.0, oskar_mem_double(
                oskar_station_element_y_weights(station), &status)[i], tol);
        EXPECT_NEAR(i/14.0, oskar_mem_double(
                oskar_station_element_z_weights(station), &status)[i], tol);
        EXPECT_NEAR(1.0, oskar_mem_double2(
                oskar_station_element_weight(station), &status)[i].x, tol);
        EXPECT_NEAR(0.0, oskar_mem_double2(
                oskar_station_element_weight(station), &status)[i].y, tol);
        EXPECT_NEAR(1.0, oskar_mem_double(
                oskar_station_element_gain(station), &status)[i], tol);
        EXPECT_NEAR(0.0, oskar_mem_double(
                oskar_station_element_gain_error(station), &status)[i], tol);
        EXPECT_NEAR(0.0, oskar_mem_double(
                oskar_station_element_phase_offset(station), &status)[i], tol);
        EXPECT_NEAR(0.0, oskar_mem_double(
                oskar_station_element_phase_error(station), &status)[i], tol);
    }

    // Remove the test file.
    remove(filename);

    // Free memory.
    oskar_station_free(station, &status);
}

