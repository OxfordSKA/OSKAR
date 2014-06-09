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

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <oskar_convert_enu_to_offset_ecef.h>
#include <oskar_telescope.h>
#include <oskar_station_load_config.h>
#include <oskar_get_error_string.h>
#include <oskar_mem.h>
#include <oskar_timer.h>

#include <cmath>
#include <cstdio>

static const char* telescope_file_name = "test_telescope.dat";
static const char* station_base = "test_station";
static const int n_stations = 25;
static const int n_elements = 200;

static void create_test_data()
{
    // Create a telescope coordinate file.
    FILE* file;
    char station_name[80];
    file = fopen(telescope_file_name, "w");
    for (int i = 0; i < n_stations; ++i)
        fprintf(file, "%.8f,%.8f,%.8f\n", i / 10.0, i / 20.0, i / 30.0);
    fclose(file);

    // Create some station coordinate files.
    for (int i = 0; i < n_stations; ++i)
    {
        sprintf(station_name, "%s_%d.dat", station_base, i);
        file = fopen(station_name, "w");
        for (int j = 0; j < n_elements; ++j)
        {
            int t = j + i;
            fprintf(file, "%.8f,%.8f,%.8f\n", t / 5.0, t / 6.0, t / 7.0);
        }
        fclose(file);
    }
}

static void delete_test_data()
{
    char station_name[80];

    // Remove telescope coordinate file.
    remove(telescope_file_name);

    // Remove station coordinate files.
    for (int i = 0; i < n_stations; ++i)
    {
        sprintf(station_name, "%s_%d.dat", station_base, i);
        remove(station_name);
    }
}

TEST(TelescopeModel, load_telescope_cpu)
{
    create_test_data();

    char station_name[80];
    int status = 0;

    // Set the location.
    double longitude = 30.0 * M_PI / 180.0;
    double latitude = 50.0 * M_PI / 180.0;
    double altitude = 0.0;
    oskar_Telescope *tel_cpu, *tel_cpu2, *tel_gpu;
    tel_cpu = oskar_telescope_create(OSKAR_DOUBLE,
            OSKAR_CPU, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Fill the telescope structure.
    oskar_telescope_load_station_coords_horizon(tel_cpu, telescope_file_name,
            longitude, latitude, altitude, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n_stations; ++i)
    {
        oskar_Station* s = oskar_telescope_station(tel_cpu, i);
        sprintf(station_name, "%s_%d.dat", station_base, i);
        oskar_station_load_config(s, station_name, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        EXPECT_EQ(n_elements, oskar_station_num_elements(s));
    }

    // Copy telescope structure to GPU.
    tel_gpu = oskar_telescope_create_copy(tel_cpu,
            OSKAR_GPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Copy the telescope structure back to the CPU.
    tel_cpu2 = oskar_telescope_create_copy(tel_gpu,
            OSKAR_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    double *station_x, *station_y, *station_z;
    station_x = oskar_mem_double(
            oskar_telescope_station_true_x_offset_ecef_metres(tel_cpu2), &status);
    station_y = oskar_mem_double(
            oskar_telescope_station_true_y_offset_ecef_metres(tel_cpu2), &status);
    station_z = oskar_mem_double(
            oskar_telescope_station_true_z_offset_ecef_metres(tel_cpu2), &status);

    // Check the contents of the CPU structure.
    for (int i = 0; i < n_stations; ++i)
    {
        oskar_Station* s = oskar_telescope_station(tel_cpu2, i);

        // Define horizon coordinates.
        double hor_x = i / 10.0;
        double hor_y = i / 20.0;
        double hor_z = i / 30.0;

        // Compute offset geocentric coordinates.
        double x = 0.0, y = 0.0, z = 0.0;
        oskar_convert_enu_to_offset_ecef_d(1, &hor_x, &hor_y, &hor_z,
                longitude, latitude, &x, &y, &z);
        EXPECT_NEAR(x, station_x[i], 1e-5);
        EXPECT_NEAR(y, station_y[i], 1e-5);
        EXPECT_NEAR(z, station_z[i], 1e-5);

        double *e_x, *e_y, *e_z;
        e_x = oskar_mem_double(
                oskar_station_element_measured_x_enu_metres(s), &status);
        e_y = oskar_mem_double(
                oskar_station_element_measured_y_enu_metres(s), &status);
        e_z = oskar_mem_double(
                oskar_station_element_measured_z_enu_metres(s), &status);

        for (int j = 0; j < n_elements; ++j)
        {
            int t = j + i;
            double x = t / 5.0;
            double y = t / 6.0;
            double z = t / 7.0;
            EXPECT_NEAR(x, e_x[j], 1e-5);
            EXPECT_NEAR(y, e_y[j], 1e-5);
            EXPECT_NEAR(z, e_z[j], 1e-5);
        }
    }

    // Free memory.
    oskar_telescope_free(tel_cpu, &status);
    oskar_telescope_free(tel_cpu2, &status);
    oskar_telescope_free(tel_gpu, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    delete_test_data();
}
