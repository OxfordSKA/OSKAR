/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <cstdio>
#include <fstream>

#include <gtest/gtest.h>

#include "math/oskar_cmath.h"
#include "interferometer/oskar_jones.h"
#include "interferometer/oskar_jones_apply_cable_length_errors.h"
#include "telescope/oskar_telescope.h"
#include "utility/oskar_device_count.h"
#include "utility/oskar_dir.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_vector_types.h"

#define C_0 299792458.0

using std::string;

namespace { // Begin anonymous namespace for file-local utility functions.


double check_gpu_version(
        const oskar_Jones* jones,
        const oskar_Telescope* tel,
        double frequency_hz,
        int* status
)
{
    double max_rel_error = 0.0;
    int location = 0;
    const int num_devices = oskar_device_count(0, &location);
    if (num_devices > 0)
    {
        // Copy the telescope model to the device.
        oskar_Telescope* tel_device = oskar_telescope_create_copy(
                tel, location, status
        );

        // Create a test Jones matrix block and fill it with ones.
        oskar_Jones* jones_device = oskar_jones_create(
                oskar_jones_type(jones), location,
                oskar_jones_num_stations(jones),
                oskar_jones_num_sources(jones), status
        );
        oskar_Mem* mem_device = oskar_jones_mem(jones_device);
        oskar_mem_set_value_real(
                mem_device,
                1.0, 0, oskar_mem_length(oskar_jones_mem_const(jones)), status
        );

        // Apply the cable length errors.
        const oskar_Mem* error[] =
        {
            oskar_telescope_station_cable_length_error_const(tel_device, 0),
            oskar_telescope_station_cable_length_error_const(tel_device, 1)
        };
        oskar_jones_apply_cable_length_errors(
                jones_device, frequency_hz, error[0], error[1], status
        );

        // Check results are consistent.
        oskar_mem_evaluate_relative_error(
                mem_device, oskar_jones_mem_const(jones),
                0, &max_rel_error, 0, 0, status
        );

        // Clean up.
        oskar_jones_free(jones_device, status);
        oskar_telescope_free(tel_device, status);
    }
    return max_rel_error;
}


void set_ones(oskar_Jones* jones, int* status)
{
    oskar_mem_set_value_real(
            oskar_jones_mem(jones),
            1.0, 0, oskar_mem_length(oskar_jones_mem(jones)), status
    );
}


void write_cable_length_error(
        const string& filename, int num, double factor, double offset
)
{
    FILE* file = fopen(filename.c_str(), "w");
    for (int i = 0; i < num; ++i)
    {
        (void) fprintf(file, "%.3f\n", i * factor + offset);
    }
    (void) fclose(file);
}


void write_layout(const string& filename, int num, double factor)
{
    FILE* file = fopen(filename.c_str(), "w");
    for (int i = 0; i < num; ++i)
    {
        (void) fprintf(
                file, "%.3f, %.3f, %.3f\n",
                i * (factor + 0.1), i * (factor + 0.2), i * (factor + 0.3)
        );
    }
    (void) fclose(file);
}

} // End anonymous namespace for file-local utility functions.


TEST(jones_apply_cable_length_errors, test_different_matrix)
{
    // Define telescope and station model sizes.
    const char* tel_name = (
            "temp_test_telescope_apply_cable_error_different_matrix.tm"
    );
    const int num_sources = 12;
    const int num_stations = 5;
    const double frequency_hz = 100e6;
    const double k = 2 * M_PI * frequency_hz / C_0;

    // Create a telescope model to load.
    {
        const string tel_dir = string(tel_name) + oskar_dir_separator();
        ASSERT_EQ(1, oskar_dir_mkdir(tel_dir.c_str()));

        // Create the telescope position and layout files.
        std::ofstream position(tel_dir + "position.txt");
        position << "116.2, -26.1, 123.4\n";
        write_layout(tel_dir + "layout.txt", num_stations, 1);

        // Create the telescope cable length error files.
        write_cable_length_error(
                tel_dir + "cable_length_error_x.txt", num_stations, 3.14, 2.718
        );
        write_cable_length_error(
                tel_dir + "cable_length_error_y.txt", num_stations, 2.718, 3.14
        );
    }

    // Load the telescope model in both single and double precision.
    const int prec[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    for (int i = 0; i < 2; ++i)
    {
        int status = 0;

        // Create an empty telescope model in memory.
        oskar_Telescope* tel = oskar_telescope_create(
                prec[i], OSKAR_CPU, 0, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Load the telescope model from the directory.
        oskar_telescope_load(tel, tel_name, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Create a test Jones matrix block and fill it with ones.
        oskar_Jones* jones = oskar_jones_create(
                prec[i] | OSKAR_COMPLEX | OSKAR_MATRIX, OSKAR_CPU,
                num_stations, num_sources, &status
        );
        set_ones(jones, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Apply the cable length errors.
        const oskar_Mem* cable_error[] =
        {
            oskar_telescope_station_cable_length_error_const(tel, 0),
            oskar_telescope_station_cable_length_error_const(tel, 1)
        };
        oskar_jones_apply_cable_length_errors(
                jones, frequency_hz, cable_error[0], cable_error[1], &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check the values were applied correctly.
        const double tol = (prec[i] == OSKAR_DOUBLE) ? 1e-12 : 1e-6;
        if (prec[i] == OSKAR_DOUBLE)
        {
            double4c* data = oskar_mem_double4c(
                    oskar_jones_mem(jones), &status
            );
            const double* error[] =
            {
                oskar_mem_double_const(cable_error[0], &status),
                oskar_mem_double_const(cable_error[1], &status)
            };
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            for (int p = 0; p < num_stations; ++p)
            {
                for (int s = 0; s < num_sources; ++s)
                {
                    const int idx = p * num_sources + s;
                    EXPECT_NEAR(cos(k * error[0][p]), data[idx].a.x, tol);
                    EXPECT_NEAR(sin(k * error[0][p]), data[idx].a.y, tol);
                    EXPECT_EQ(0.0, data[idx].b.x);
                    EXPECT_EQ(0.0, data[idx].b.y);
                    EXPECT_EQ(0.0, data[idx].c.x);
                    EXPECT_EQ(0.0, data[idx].c.y);
                    EXPECT_NEAR(cos(k * error[1][p]), data[idx].d.x, tol);
                    EXPECT_NEAR(sin(k * error[1][p]), data[idx].d.y, tol);
                }
            }
        }
        else
        {
            float4c* data = oskar_mem_float4c(
                    oskar_jones_mem(jones), &status
            );
            const float* error[] =
            {
                oskar_mem_float_const(cable_error[0], &status),
                oskar_mem_float_const(cable_error[1], &status)
            };
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            for (int p = 0; p < num_stations; ++p)
            {
                for (int s = 0; s < num_sources; ++s)
                {
                    const int idx = p * num_sources + s;
                    EXPECT_NEAR(cos(k * error[0][p]), data[idx].a.x, tol);
                    EXPECT_NEAR(sin(k * error[0][p]), data[idx].a.y, tol);
                    EXPECT_EQ(0.0, data[idx].b.x);
                    EXPECT_EQ(0.0, data[idx].b.y);
                    EXPECT_EQ(0.0, data[idx].c.x);
                    EXPECT_EQ(0.0, data[idx].c.y);
                    EXPECT_NEAR(cos(k * error[1][p]), data[idx].d.x, tol);
                    EXPECT_NEAR(sin(k * error[1][p]), data[idx].d.y, tol);
                }
            }
        }

        // Check the GPU version, if applicable.
        ASSERT_LT(check_gpu_version(jones, tel, frequency_hz, &status), tol);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Clean up.
        oskar_jones_free(jones, &status);
        oskar_telescope_free(tel, &status);
    }
    oskar_dir_remove(tel_name);
}


TEST(jones_apply_cable_length_errors, test_different_scalar)
{
    // Define telescope and station model sizes.
    const char* tel_name = (
            "temp_test_telescope_apply_cable_error_different_scalar.tm"
    );
    const int num_sources = 12;
    const int num_stations = 5;
    const double frequency_hz = 100e6;
    const double k = 2 * M_PI * frequency_hz / C_0;

    // Create a telescope model to load.
    {
        const string tel_dir = string(tel_name) + oskar_dir_separator();
        ASSERT_EQ(1, oskar_dir_mkdir(tel_dir.c_str()));

        // Create the telescope position and layout files.
        std::ofstream position(tel_dir + "position.txt");
        position << "116.2, -26.1, 123.4\n";
        write_layout(tel_dir + "layout.txt", num_stations, 1);

        // Create the telescope cable length error files.
        write_cable_length_error(
                tel_dir + "cable_length_error_x.txt", num_stations, 3.14, 2.718
        );
        write_cable_length_error(
                tel_dir + "cable_length_error_y.txt", num_stations, 2.718, 3.14
        );
    }

    // Load the telescope model in both single and double precision.
    const int prec[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    for (int i = 0; i < 2; ++i)
    {
        int status = 0;

        // Create an empty telescope model in memory.
        oskar_Telescope* tel = oskar_telescope_create(
                prec[i], OSKAR_CPU, 0, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Load the telescope model from the directory.
        oskar_telescope_load(tel, tel_name, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Create a test Jones matrix block and fill it with ones.
        oskar_Jones* jones = oskar_jones_create(
                prec[i] | OSKAR_COMPLEX, OSKAR_CPU,
                num_stations, num_sources, &status
        );
        set_ones(jones, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Apply the cable length errors.
        const oskar_Mem* cable_error[] =
        {
            oskar_telescope_station_cable_length_error_const(tel, 0),
            oskar_telescope_station_cable_length_error_const(tel, 1)
        };
        oskar_jones_apply_cable_length_errors(
                jones, frequency_hz, cable_error[0], cable_error[1], &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check the values were applied correctly.
        const double tol = (prec[i] == OSKAR_DOUBLE) ? 1e-12 : 1e-6;
        if (prec[i] == OSKAR_DOUBLE)
        {
            double2* data = oskar_mem_double2(oskar_jones_mem(jones), &status);
            const double* error[] =
            {
                oskar_mem_double_const(cable_error[0], &status),
                oskar_mem_double_const(cable_error[1], &status)
            };
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            for (int p = 0; p < num_stations; ++p)
            {
                for (int s = 0; s < num_sources; ++s)
                {
                    const int idx = p * num_sources + s;
                    EXPECT_NEAR(cos(k * error[0][p]), data[idx].x, tol);
                    EXPECT_NEAR(sin(k * error[0][p]), data[idx].y, tol);
                }
            }
        }
        else
        {
            float2* data = oskar_mem_float2(oskar_jones_mem(jones), &status);
            const float* error[] =
            {
                oskar_mem_float_const(cable_error[0], &status),
                oskar_mem_float_const(cable_error[1], &status)
            };
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            for (int p = 0; p < num_stations; ++p)
            {
                for (int s = 0; s < num_sources; ++s)
                {
                    const int idx = p * num_sources + s;
                    EXPECT_NEAR(cos(k * error[0][p]), data[idx].x, tol);
                    EXPECT_NEAR(sin(k * error[0][p]), data[idx].y, tol);
                }
            }
        }

        // Check the GPU version, if applicable.
        ASSERT_LT(check_gpu_version(jones, tel, frequency_hz, &status), tol);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Clean up.
        oskar_jones_free(jones, &status);
        oskar_telescope_free(tel, &status);
    }
    oskar_dir_remove(tel_name);
}


TEST(jones_apply_cable_length_errors, test_same_matrix)
{
    // Define telescope and station model sizes.
    const char* tel_name = (
            "temp_test_telescope_apply_cable_error_same_matrix.tm"
    );
    const int num_sources = 12;
    const int num_stations = 5;
    const double frequency_hz = 100e6;
    const double k = 2 * M_PI * frequency_hz / C_0;

    // Create a telescope model to load.
    {
        const string tel_dir = string(tel_name) + oskar_dir_separator();
        ASSERT_EQ(1, oskar_dir_mkdir(tel_dir.c_str()));

        // Create the telescope position and layout files.
        std::ofstream position(tel_dir + "position.txt");
        position << "116.2, -26.1, 123.4\n";
        write_layout(tel_dir + "layout.txt", num_stations, 1);

        // Create the telescope cable length error file.
        write_cable_length_error(
                tel_dir + "cable_length_error.txt", num_stations, 3.14, 2.718
        );
    }

    // Load the telescope model in both single and double precision.
    const int prec[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    for (int i = 0; i < 2; ++i)
    {
        int status = 0;

        // Create an empty telescope model in memory.
        oskar_Telescope* tel = oskar_telescope_create(
                prec[i], OSKAR_CPU, 0, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Load the telescope model from the directory.
        oskar_telescope_load(tel, tel_name, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Create a test Jones matrix block and fill it with ones.
        oskar_Jones* jones = oskar_jones_create(
                prec[i] | OSKAR_COMPLEX | OSKAR_MATRIX, OSKAR_CPU,
                num_stations, num_sources, &status
        );
        set_ones(jones, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Apply the cable length errors.
        const oskar_Mem* cable_error[] =
        {
            oskar_telescope_station_cable_length_error_const(tel, 0),
            oskar_telescope_station_cable_length_error_const(tel, 1)
        };
        ASSERT_TRUE(cable_error[0]);
        ASSERT_TRUE(cable_error[1]);
        oskar_jones_apply_cable_length_errors(
                jones, frequency_hz, cable_error[0], cable_error[1], &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check the values were applied correctly.
        const double tol = (prec[i] == OSKAR_DOUBLE) ? 1e-12 : 1e-6;
        if (prec[i] == OSKAR_DOUBLE)
        {
            double4c* data = oskar_mem_double4c(
                    oskar_jones_mem(jones), &status
            );
            const double* error[] =
            {
                oskar_mem_double_const(cable_error[0], &status),
                oskar_mem_double_const(cable_error[1], &status)
            };
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            for (int p = 0; p < num_stations; ++p)
            {
                for (int s = 0; s < num_sources; ++s)
                {
                    const int idx = p * num_sources + s;
                    EXPECT_NEAR(cos(k * error[0][p]), data[idx].a.x, tol);
                    EXPECT_NEAR(sin(k * error[0][p]), data[idx].a.y, tol);
                    EXPECT_EQ(0.0, data[idx].b.x);
                    EXPECT_EQ(0.0, data[idx].b.y);
                    EXPECT_EQ(0.0, data[idx].c.x);
                    EXPECT_EQ(0.0, data[idx].c.y);
                    EXPECT_NEAR(cos(k * error[1][p]), data[idx].d.x, tol);
                    EXPECT_NEAR(sin(k * error[1][p]), data[idx].d.y, tol);
                }
            }
        }
        else
        {
            float4c* data = oskar_mem_float4c(
                    oskar_jones_mem(jones), &status
            );
            const float* error[] =
            {
                oskar_mem_float_const(cable_error[0], &status),
                oskar_mem_float_const(cable_error[1], &status)
            };
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            for (int p = 0; p < num_stations; ++p)
            {
                for (int s = 0; s < num_sources; ++s)
                {
                    const int idx = p * num_sources + s;
                    EXPECT_NEAR(cos(k * error[0][p]), data[idx].a.x, tol);
                    EXPECT_NEAR(sin(k * error[0][p]), data[idx].a.y, tol);
                    EXPECT_EQ(0.0, data[idx].b.x);
                    EXPECT_EQ(0.0, data[idx].b.y);
                    EXPECT_EQ(0.0, data[idx].c.x);
                    EXPECT_EQ(0.0, data[idx].c.y);
                    EXPECT_NEAR(cos(k * error[1][p]), data[idx].d.x, tol);
                    EXPECT_NEAR(sin(k * error[1][p]), data[idx].d.y, tol);
                }
            }
        }

        // Check the GPU version, if applicable.
        ASSERT_LT(check_gpu_version(jones, tel, frequency_hz, &status), tol);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Clean up.
        oskar_jones_free(jones, &status);
        oskar_telescope_free(tel, &status);
    }
    oskar_dir_remove(tel_name);
}


TEST(jones_apply_cable_length_errors, test_same_scalar)
{
    // Define telescope and station model sizes.
    const char* tel_name = (
            "temp_test_telescope_apply_cable_error_same_scalar.tm"
    );
    const int num_sources = 12;
    const int num_stations = 5;
    const double frequency_hz = 100e6;
    const double k = 2 * M_PI * frequency_hz / C_0;

    // Create a telescope model to load.
    {
        const string tel_dir = string(tel_name) + oskar_dir_separator();
        ASSERT_EQ(1, oskar_dir_mkdir(tel_dir.c_str()));

        // Create the telescope position and layout files.
        std::ofstream position(tel_dir + "position.txt");
        position << "116.2, -26.1, 123.4\n";
        write_layout(tel_dir + "layout.txt", num_stations, 1);

        // Create the telescope cable length error file.
        write_cable_length_error(
                tel_dir + "cable_length_error.txt", num_stations, 3.14, 2.718
        );
    }

    // Load the telescope model in both single and double precision.
    const int prec[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    for (int i = 0; i < 2; ++i)
    {
        int status = 0;

        // Create an empty telescope model in memory.
        oskar_Telescope* tel = oskar_telescope_create(
                prec[i], OSKAR_CPU, 0, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Load the telescope model from the directory.
        oskar_telescope_load(tel, tel_name, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Create a test Jones matrix block and fill it with ones.
        oskar_Jones* jones = oskar_jones_create(
                prec[i] | OSKAR_COMPLEX, OSKAR_CPU,
                num_stations, num_sources, &status
        );
        set_ones(jones, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Apply the cable length errors.
        const oskar_Mem* cable_error[] =
        {
            oskar_telescope_station_cable_length_error_const(tel, 0),
            oskar_telescope_station_cable_length_error_const(tel, 1)
        };
        ASSERT_TRUE(cable_error[0]);
        ASSERT_TRUE(cable_error[1]);
        oskar_jones_apply_cable_length_errors(
                jones, frequency_hz, cable_error[0], cable_error[1], &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check the values were applied correctly.
        const double tol = (prec[i] == OSKAR_DOUBLE) ? 1e-12 : 1e-6;
        if (prec[i] == OSKAR_DOUBLE)
        {
            double2* data = oskar_mem_double2(oskar_jones_mem(jones), &status);
            const double* error[] =
            {
                oskar_mem_double_const(cable_error[0], &status),
                oskar_mem_double_const(cable_error[1], &status)
            };
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            for (int p = 0; p < num_stations; ++p)
            {
                for (int s = 0; s < num_sources; ++s)
                {
                    const int idx = p * num_sources + s;
                    EXPECT_NEAR(cos(k * error[0][p]), data[idx].x, tol);
                    EXPECT_NEAR(sin(k * error[0][p]), data[idx].y, tol);
                }
            }
        }
        else
        {
            float2* data = oskar_mem_float2(oskar_jones_mem(jones), &status);
            const float* error[] =
            {
                oskar_mem_float_const(cable_error[0], &status),
                oskar_mem_float_const(cable_error[1], &status)
            };
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            for (int p = 0; p < num_stations; ++p)
            {
                for (int s = 0; s < num_sources; ++s)
                {
                    const int idx = p * num_sources + s;
                    EXPECT_NEAR(cos(k * error[0][p]), data[idx].x, tol);
                    EXPECT_NEAR(sin(k * error[0][p]), data[idx].y, tol);
                }
            }
        }

        // Check the GPU version, if applicable.
        ASSERT_LT(check_gpu_version(jones, tel, frequency_hz, &status), tol);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Clean up.
        oskar_jones_free(jones, &status);
        oskar_telescope_free(tel, &status);
    }
    oskar_dir_remove(tel_name);
}
