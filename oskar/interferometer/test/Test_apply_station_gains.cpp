/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <cstdio>
#include <fstream>

#include <gtest/gtest.h>

#include "math/oskar_cmath.h"
#include "interferometer/oskar_jones.h"
#include "interferometer/oskar_jones_apply_station_gains.h"
#include "telescope/oskar_telescope.h"
#include "utility/oskar_device_count.h"
#include "utility/oskar_dir.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_hdf5.h"
#include "utility/oskar_vector_types.h"

#define C_0 299792458.0
#define INDEX_3D(N3, N2, N1, I3, I2, I1)  (N1 * (N2 * I3 + I2) + I1)

using std::string;

namespace { // Begin anonymous namespace for file-local utility functions.


double check_gpu_version(
        const oskar_Jones* jones,
        const oskar_Telescope* tel,
        int time_idx,
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

        // Evaluate the station gains.
        oskar_Mem* gains = oskar_mem_create(
                oskar_jones_type(jones), location,
                oskar_jones_num_stations(jones), status
        );
        oskar_gains_evaluate(
                oskar_telescope_gains_const(tel), time_idx, frequency_hz,
                gains, 0, status
        );

        // Create a test Jones matrix block and fill it with something.
        oskar_Jones* jones_device = oskar_jones_create(
                oskar_jones_type(jones), location,
                oskar_jones_num_stations(jones),
                oskar_jones_num_sources(jones), status
        );
        oskar_Mem* mem_device = oskar_jones_mem(jones_device);
        oskar_mem_set_value_real(
                mem_device, 2.0, 0, oskar_mem_length(mem_device), status
        );

        // Apply the station gains.
        oskar_jones_apply_station_gains(jones_device, gains, status);

        // Check results are consistent.
        oskar_mem_evaluate_relative_error(
                mem_device, oskar_jones_mem_const(jones),
                0, &max_rel_error, 0, 0, status
        );

        // Clean up.
        oskar_mem_free(gains, status);
        oskar_jones_free(jones_device, status);
        oskar_telescope_free(tel_device, status);
    }
    return max_rel_error;
}


void write_gain_model(
        const string& filename,
        int num_times,
        int num_channels,
        int num_antennas,
        int num_pols,
        int* status
)
{
    oskar_Mem *freq_hz = 0, *gains_x = 0, *gains_y = 0;
    oskar_HDF5* gain_file = oskar_hdf5_open(filename.c_str(), 'w', status);
    freq_hz = oskar_mem_create(
            OSKAR_DOUBLE, OSKAR_CPU, num_channels, status
    );
    for (size_t i = 0; i < oskar_mem_length(freq_hz); ++i)
    {
        oskar_mem_set_element_real(freq_hz, i, 100e6 + i * 1e6, status);
    }
    oskar_hdf5_write_dataset(
            gain_file, 0, "freq (Hz)", 1, 0, freq_hz, 0, status
    );
    const size_t num_elements = num_times * num_channels * num_antennas;
    const size_t dims[] =
    {
        (size_t) num_times,
        (size_t) num_channels,
        (size_t) num_antennas
    };
    gains_x = oskar_mem_create(
            OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, num_elements, status
    );
    gains_y = oskar_mem_create(
            OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, num_elements, status
    );
    for (int t = 0; t < num_times; ++t)
    {
        for (int c = 0; c < num_channels; ++c)
        {
            for (int a = 0; a < num_antennas; ++a)
            {
                const size_t idx = INDEX_3D(
                        num_times, num_channels, num_antennas, t, c, a
                );
                const double value_x = 100000. * t + 1000. * c + a;
                const double value_y = 200000. * t + 2000. * c + a;
                oskar_mem_set_element_real(gains_x, 2 * idx, value_x, status);
                oskar_mem_set_element_real(gains_y, 2 * idx, value_y, status);
            }
        }
    }
    const int num_dims_to_write = 3;
    oskar_hdf5_write_dataset(
            gain_file, 0, "gain_xpol",
            num_dims_to_write, dims, gains_x, 0, status
    );
    if (num_pols > 1)
    {
        oskar_hdf5_write_dataset(
                gain_file, 0, "gain_ypol",
                num_dims_to_write, dims, gains_y, 0, status
        );
    }

    // Clean up.
    oskar_mem_free(gains_x, status);
    oskar_mem_free(gains_y, status);
    oskar_mem_free(freq_hz, status);
    oskar_hdf5_close(gain_file);
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


TEST(jones_apply_station_gains, test_different_matrix)
{
    // Define telescope and station model sizes.
    const char* tel_name = (
            "temp_test_telescope_apply_station_gains_different_matrix.tm"
    );
    const int num_sources = 12;
    const int num_stations = 5;
    const int num_channels = 16;
    const int num_times = 8;

    // Create a telescope model to load.
    {
        int status = 0;
        const string tel_dir = string(tel_name) + oskar_dir_separator();
        ASSERT_EQ(1, oskar_dir_mkdir(tel_dir.c_str()));

        // Create the telescope position and layout files.
        std::ofstream position(tel_dir + "position.txt");
        position << "116.2, -26.1, 123.4\n";
        write_layout(tel_dir + "layout.txt", num_stations, 1);

        // Create the gain model.
        write_gain_model(
                tel_dir + "gain_model.h5",
                num_times, num_channels, num_stations, 2, &status
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

        // Evaluate the station gains.
        const int time_idx = 6;
        const int chan_idx = 8;
        const double frequency_hz = 100e6 + chan_idx * 1e6;
        oskar_Mem* gains = oskar_mem_create(
                prec[i] | OSKAR_COMPLEX | OSKAR_MATRIX,
                OSKAR_CPU, num_stations, &status
        );
        oskar_gains_evaluate(
                oskar_telescope_gains_const(tel), time_idx, frequency_hz,
                gains, 0, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check the gains were evaluated correctly.
        const double tol = (prec[i] == OSKAR_DOUBLE) ? 1e-12 : 1e-6;
        if (prec[i] == OSKAR_SINGLE)
        {
            const float4c* ptr = oskar_mem_float4c_const(gains, &status);
            for (int a = 0; a < num_stations; ++a)
            {
                ASSERT_NEAR(
                        100000. * time_idx + 1000. * chan_idx + a,
                        ptr[a].a.x, tol
                );
                ASSERT_NEAR(
                        200000. * time_idx + 2000. * chan_idx + a,
                        ptr[a].d.x, tol
                );
                ASSERT_DOUBLE_EQ(0.0, ptr[a].a.y);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].d.y);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].b.x);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].b.y);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].c.x);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].c.y);
            }
        }
        else
        {
            const double4c* ptr = oskar_mem_double4c_const(gains, &status);
            for (int a = 0; a < num_stations; ++a)
            {
                ASSERT_NEAR(
                        100000. * time_idx + 1000. * chan_idx + a,
                        ptr[a].a.x, tol
                );
                ASSERT_NEAR(
                        200000. * time_idx + 2000. * chan_idx + a,
                        ptr[a].d.x, tol
                );
                ASSERT_DOUBLE_EQ(0.0, ptr[a].a.y);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].d.y);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].b.x);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].b.y);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].c.x);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].c.y);
            }
        }

        // Create a test Jones matrix block and fill it with something.
        oskar_Jones* jones = oskar_jones_create(
                prec[i] | OSKAR_COMPLEX | OSKAR_MATRIX, OSKAR_CPU,
                num_stations, num_sources, &status
        );
        oskar_mem_set_value_real(
                oskar_jones_mem(jones),
                2.0, 0, oskar_mem_length(oskar_jones_mem(jones)), &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Apply the station gains.
        oskar_jones_apply_station_gains(jones, gains, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check the gains were applied correctly.
        if (prec[i] == OSKAR_SINGLE)
        {
            const float4c* ptr = oskar_mem_float4c_const(
                    oskar_jones_mem_const(jones), &status
            );
            for (int a = 0; a < num_stations; ++a)
            {
                for (int s = 0; s < num_sources; ++s)
                {
                    const size_t idx = a * num_sources + s;
                    ASSERT_NEAR(
                            2 * (100000. * time_idx + 1000. * chan_idx + a),
                            ptr[idx].a.x, tol
                    );
                    ASSERT_NEAR(
                            2 * (200000. * time_idx + 2000. * chan_idx + a),
                            ptr[idx].d.x, tol
                    );
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].a.y);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].d.y);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].b.x);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].b.y);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].c.x);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].c.y);
                }
            }
        }
        else
        {
            const double4c* ptr = oskar_mem_double4c_const(
                    oskar_jones_mem_const(jones), &status
            );
            for (int a = 0; a < num_stations; ++a)
            {
                for (int s = 0; s < num_sources; ++s)
                {
                    const size_t idx = a * num_sources + s;
                    ASSERT_NEAR(
                            2 * (100000. * time_idx + 1000. * chan_idx + a),
                            ptr[idx].a.x, tol
                    );
                    ASSERT_NEAR(
                            2 * (200000. * time_idx + 2000. * chan_idx + a),
                            ptr[idx].d.x, tol
                    );
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].a.y);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].d.y);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].b.x);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].b.y);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].c.x);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].c.y);
                }
            }
        }
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check the GPU version, if applicable.
        ASSERT_LT(
                check_gpu_version(
                        jones, tel, time_idx, frequency_hz, &status
                ),
                tol
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Clean up.
        oskar_mem_free(gains, &status);
        oskar_jones_free(jones, &status);
        oskar_telescope_free(tel, &status);
    }
    oskar_dir_remove(tel_name);
}


TEST(jones_apply_station_gains, test_different_scalar)
{
    // Define telescope and station model sizes.
    const char* tel_name = (
            "temp_test_telescope_apply_station_gains_different_scalar.tm"
    );
    const int num_sources = 12;
    const int num_stations = 5;
    const int num_channels = 16;
    const int num_times = 8;

    // Create a telescope model to load.
    {
        int status = 0;
        const string tel_dir = string(tel_name) + oskar_dir_separator();
        ASSERT_EQ(1, oskar_dir_mkdir(tel_dir.c_str()));

        // Create the telescope position and layout files.
        std::ofstream position(tel_dir + "position.txt");
        position << "116.2, -26.1, 123.4\n";
        write_layout(tel_dir + "layout.txt", num_stations, 1);

        // Create the gain model.
        write_gain_model(
                tel_dir + "gain_model.h5",
                num_times, num_channels, num_stations, 2, &status
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

        // Evaluate the station gains.
        const int time_idx = 6;
        const int chan_idx = 8;
        const double frequency_hz = 100e6 + chan_idx * 1e6;
        oskar_Mem* gains_x = oskar_mem_create(
                prec[i] | OSKAR_COMPLEX, OSKAR_CPU, num_stations, &status
        );
        oskar_Mem* gains_y = oskar_mem_create(
                prec[i] | OSKAR_COMPLEX, OSKAR_CPU, num_stations, &status
        );
        oskar_gains_evaluate(
                oskar_telescope_gains_const(tel), time_idx, frequency_hz,
                gains_x, 0, &status
        );
        oskar_gains_evaluate(
                oskar_telescope_gains_const(tel), time_idx, frequency_hz,
                gains_y, 1, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check the gains were evaluated correctly.
        const double tol = (prec[i] == OSKAR_DOUBLE) ? 1e-12 : 1e-6;
        if (prec[i] == OSKAR_SINGLE)
        {
            const float2* ptr_x = oskar_mem_float2_const(gains_x, &status);
            const float2* ptr_y = oskar_mem_float2_const(gains_y, &status);
            for (int a = 0; a < num_stations; ++a)
            {
                ASSERT_NEAR(
                        100000. * time_idx + 1000. * chan_idx + a,
                        ptr_x[a].x, tol
                );
                ASSERT_NEAR(
                        200000. * time_idx + 2000. * chan_idx + a,
                        ptr_y[a].x, tol
                );
            }
        }
        else
        {
            const double2* ptr_x = oskar_mem_double2_const(gains_x, &status);
            const double2* ptr_y = oskar_mem_double2_const(gains_y, &status);
            for (int a = 0; a < num_stations; ++a)
            {
                ASSERT_NEAR(
                        100000. * time_idx + 1000. * chan_idx + a,
                        ptr_x[a].x, tol
                );
                ASSERT_NEAR(
                        200000. * time_idx + 2000. * chan_idx + a,
                        ptr_y[a].x, tol
                );
            }
        }

        // Create a test Jones matrix block and fill it with something.
        oskar_Jones* jones = oskar_jones_create(
                prec[i] | OSKAR_COMPLEX, OSKAR_CPU,
                num_stations, num_sources, &status
        );
        oskar_mem_set_value_real(
                oskar_jones_mem(jones),
                2.0, 0, oskar_mem_length(oskar_jones_mem(jones)), &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Apply the station gains.
        oskar_jones_apply_station_gains(jones, gains_x, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check the gains were applied correctly.
        if (prec[i] == OSKAR_SINGLE)
        {
            const float2* ptr = oskar_mem_float2_const(
                    oskar_jones_mem_const(jones), &status
            );
            for (int a = 0; a < num_stations; ++a)
            {
                for (int s = 0; s < num_sources; ++s)
                {
                    ASSERT_NEAR(
                            2 * (100000. * time_idx + 1000. * chan_idx + a),
                            ptr[a * num_sources + s].x, tol
                    );
                }
            }
        }
        else
        {
            const double2* ptr = oskar_mem_double2_const(
                    oskar_jones_mem_const(jones), &status
            );
            for (int a = 0; a < num_stations; ++a)
            {
                for (int s = 0; s < num_sources; ++s)
                {
                    ASSERT_NEAR(
                            2 * (100000. * time_idx + 1000. * chan_idx + a),
                            ptr[a * num_sources + s].x, tol
                    );
                }
            }
        }
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check the GPU version, if applicable.
        ASSERT_LT(
                check_gpu_version(
                        jones, tel, time_idx, frequency_hz, &status
                ),
                tol
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Clean up.
        oskar_mem_free(gains_x, &status);
        oskar_mem_free(gains_y, &status);
        oskar_jones_free(jones, &status);
        oskar_telescope_free(tel, &status);
    }
    oskar_dir_remove(tel_name);
}


TEST(jones_apply_station_gains, test_same_matrix)
{
    // Define telescope and station model sizes.
    const char* tel_name = (
            "temp_test_telescope_apply_station_gains_same_matrix.tm"
    );
    const int num_sources = 12;
    const int num_stations = 5;
    const int num_channels = 16;
    const int num_times = 8;

    // Create a telescope model to load.
    {
        int status = 0;
        const string tel_dir = string(tel_name) + oskar_dir_separator();
        ASSERT_EQ(1, oskar_dir_mkdir(tel_dir.c_str()));

        // Create the telescope position and layout files.
        std::ofstream position(tel_dir + "position.txt");
        position << "116.2, -26.1, 123.4\n";
        write_layout(tel_dir + "layout.txt", num_stations, 1);

        // Create the gain model.
        write_gain_model(
                tel_dir + "gain_model.h5",
                num_times, num_channels, num_stations, 1, &status
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

        // Evaluate the station gains.
        const int time_idx = 6;
        const int chan_idx = 8;
        const double frequency_hz = 100e6 + chan_idx * 1e6;
        oskar_Mem* gains = oskar_mem_create(
                prec[i] | OSKAR_COMPLEX | OSKAR_MATRIX,
                OSKAR_CPU, num_stations, &status
        );
        oskar_gains_evaluate(
                oskar_telescope_gains_const(tel), time_idx, frequency_hz,
                gains, 0, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check the gains were evaluated correctly.
        const double tol = (prec[i] == OSKAR_DOUBLE) ? 1e-12 : 1e-6;
        if (prec[i] == OSKAR_SINGLE)
        {
            const float4c* ptr = oskar_mem_float4c_const(gains, &status);
            for (int a = 0; a < num_stations; ++a)
            {
                ASSERT_NEAR(
                        100000. * time_idx + 1000. * chan_idx + a,
                        ptr[a].a.x, tol
                );
                ASSERT_DOUBLE_EQ(ptr[a].a.x, ptr[a].d.x);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].a.y);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].d.y);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].b.x);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].b.y);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].c.x);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].c.y);
            }
        }
        else
        {
            const double4c* ptr = oskar_mem_double4c_const(gains, &status);
            for (int a = 0; a < num_stations; ++a)
            {
                ASSERT_NEAR(
                        100000. * time_idx + 1000. * chan_idx + a,
                        ptr[a].a.x, tol
                );
                ASSERT_DOUBLE_EQ(ptr[a].a.x, ptr[a].d.x);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].a.y);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].d.y);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].b.x);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].b.y);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].c.x);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].c.y);
            }
        }

        // Create a test Jones matrix block and fill it with something.
        oskar_Jones* jones = oskar_jones_create(
                prec[i] | OSKAR_COMPLEX | OSKAR_MATRIX, OSKAR_CPU,
                num_stations, num_sources, &status
        );
        oskar_mem_set_value_real(
                oskar_jones_mem(jones),
                2.0, 0, oskar_mem_length(oskar_jones_mem(jones)), &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Apply the station gains.
        oskar_jones_apply_station_gains(jones, gains, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check the gains were applied correctly.
        if (prec[i] == OSKAR_SINGLE)
        {
            const float4c* ptr = oskar_mem_float4c_const(
                    oskar_jones_mem_const(jones), &status
            );
            for (int a = 0; a < num_stations; ++a)
            {
                for (int s = 0; s < num_sources; ++s)
                {
                    const size_t idx = a * num_sources + s;
                    ASSERT_NEAR(
                            2 * (100000. * time_idx + 1000. * chan_idx + a),
                            ptr[idx].a.x, tol
                    );
                    ASSERT_DOUBLE_EQ(ptr[idx].a.x, ptr[idx].d.x);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].a.y);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].d.y);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].b.x);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].b.y);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].c.x);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].c.y);
                }
            }
        }
        else
        {
            const double4c* ptr = oskar_mem_double4c_const(
                    oskar_jones_mem_const(jones), &status
            );
            for (int a = 0; a < num_stations; ++a)
            {
                for (int s = 0; s < num_sources; ++s)
                {
                    const size_t idx = a * num_sources + s;
                    ASSERT_NEAR(
                            2 * (100000. * time_idx + 1000. * chan_idx + a),
                            ptr[idx].a.x, tol
                    );
                    ASSERT_DOUBLE_EQ(ptr[idx].a.x, ptr[idx].d.x);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].a.y);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].d.y);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].b.x);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].b.y);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].c.x);
                    ASSERT_DOUBLE_EQ(0.0, ptr[idx].c.y);
                }
            }
        }
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check the GPU version, if applicable.
        ASSERT_LT(
                check_gpu_version(
                        jones, tel, time_idx, frequency_hz, &status
                ),
                tol
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Clean up.
        oskar_mem_free(gains, &status);
        oskar_jones_free(jones, &status);
        oskar_telescope_free(tel, &status);
    }
    oskar_dir_remove(tel_name);
}


TEST(jones_apply_station_gains, test_out_of_range)
{
    // Define telescope and station model sizes.
    const char* tel_name = (
            "temp_test_telescope_apply_station_gains_out_of_range.tm"
    );
    const int num_stations = 5;
    const int num_channels = 16;
    const int num_times = 8;

    // Create a telescope model to load.
    {
        int status = 0;
        const string tel_dir = string(tel_name) + oskar_dir_separator();
        ASSERT_EQ(1, oskar_dir_mkdir(tel_dir.c_str()));

        // Create the telescope position and layout files.
        std::ofstream position(tel_dir + "position.txt");
        position << "116.2, -26.1, 123.4\n";
        write_layout(tel_dir + "layout.txt", num_stations, 1);

        // Create the gain model.
        write_gain_model(
                tel_dir + "gain_model.h5",
                num_times, num_channels, num_stations, 1, &status
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

        // Evaluate the station gains using a time index larger than that
        // allowed by the gain table.
        // It should be clamped to the largest allowed index.
        const int chan_idx = 8;
        const double frequency_hz = 100e6 + chan_idx * 1e6;
        oskar_Mem* gains = oskar_mem_create(
                prec[i] | OSKAR_COMPLEX | OSKAR_MATRIX,
                OSKAR_CPU, num_stations, &status
        );
        oskar_gains_evaluate(
                oskar_telescope_gains_const(tel), num_times, frequency_hz,
                gains, 0, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check the gains were evaluated correctly.
        const double tol = (prec[i] == OSKAR_DOUBLE) ? 1e-12 : 1e-6;
        if (prec[i] == OSKAR_SINGLE)
        {
            const float4c* ptr = oskar_mem_float4c_const(gains, &status);
            for (int a = 0; a < num_stations; ++a)
            {
                ASSERT_NEAR(
                        100000. * (num_times - 1) + 1000. * chan_idx + a,
                        ptr[a].a.x, tol
                );
                ASSERT_DOUBLE_EQ(ptr[a].a.x, ptr[a].d.x);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].a.y);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].d.y);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].b.x);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].b.y);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].c.x);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].c.y);
            }
        }
        else
        {
            const double4c* ptr = oskar_mem_double4c_const(gains, &status);
            for (int a = 0; a < num_stations; ++a)
            {
                ASSERT_NEAR(
                        100000. * (num_times - 1) + 1000. * chan_idx + a,
                        ptr[a].a.x, tol
                );
                ASSERT_DOUBLE_EQ(ptr[a].a.x, ptr[a].d.x);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].a.y);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].d.y);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].b.x);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].b.y);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].c.x);
                ASSERT_DOUBLE_EQ(0.0, ptr[a].c.y);
            }
        }

        // Clean up.
        oskar_mem_free(gains, &status);
        oskar_telescope_free(tel, &status);
    }
    oskar_dir_remove(tel_name);
}
