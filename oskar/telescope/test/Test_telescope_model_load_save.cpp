/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "mem/oskar_mem.h"
#include "telescope/oskar_telescope.h"
#include "utility/oskar_dir.h"
#include "utility/oskar_get_error_string.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using std::vector;

static void generate_noisy_telescope(const char* dir, int num_stations,
        const vector<double>& freqs, const vector<double>& noise);

TEST(telescope_model_load_save, test_0_level)
{
    int err = 0;
    const char* tm = "temp_test_telescope_0_level";
    double longitude_rad = 0.1;
    double latitude_rad = 0.5;
    double altitude_m = 1.0;

    {
        FILE* f = 0;
        int num_stations = 10;
        char* path = 0;

        // Create a telescope model directory.
        oskar_dir_mkpath(tm);

        // Write position file.
        path = oskar_dir_get_path(tm, "position.txt");
        f = fopen(path, "w");
        fprintf(f, "0.0, 0.0\n");
        fclose(f);
        free(path);

        // Write the top-level layout file.
        path = oskar_dir_get_path(tm, "layout.txt");
        f = fopen(path, "w");
        for (int i = 0; i < num_stations; ++i)
        {
            fprintf(f, "%.1f, %.1f, %.1f\n", i * 10.0, i * 20.0, i * 30.0);
        }
        fclose(f);
        free(path);
    }

    // Load it back again.
    {
        oskar_Telescope* telescope = oskar_telescope_create(OSKAR_DOUBLE,
                        OSKAR_CPU, 0, &err);
        oskar_telescope_set_position(telescope,
                longitude_rad, latitude_rad, altitude_m);
        oskar_telescope_set_enable_numerical_patterns(telescope, 0);
        oskar_telescope_load(telescope, tm, NULL, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        oskar_telescope_free(telescope, &err);
    }

    // Remove test directory.
    oskar_dir_remove(tm);
}


TEST(telescope_model_load_save, test_1_level)
{
    int err = 0;
    const char* tm = "temp_test_telescope_1_level";

    // Create a telescope model.
    int num_stations = 10;
    int num_elements = 20;
    double longitude_rad = 0.1;
    double latitude_rad = 0.5;
    double altitude_m = 1.0;
    oskar_Telescope* telescope = oskar_telescope_create(OSKAR_SINGLE,
            OSKAR_CPU, num_stations, &err);
    oskar_telescope_resize_station_array(telescope, num_stations, &err);
    oskar_telescope_set_unique_stations(telescope, 1, &err);
    oskar_telescope_set_position(telescope, longitude_rad,
            latitude_rad, altitude_m);

    // Populate a telescope model.
    for (int i = 0; i < num_stations; ++i)
    {
        double xyz[3];
        xyz[0] = 1.0 * i;
        xyz[1] = 2.0 * i;
        xyz[2] = 3.0 * i;
        oskar_Station* st = oskar_telescope_station(telescope, i);
        oskar_telescope_set_station_coords(telescope, i, xyz,
                xyz, xyz, xyz, xyz, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        oskar_station_resize(st, num_elements, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        for (int j = 0; j < num_elements; ++j)
        {
            xyz[0] = 10.0 * i + 1.0 * j;
            xyz[1] = 20.0 * i + 1.0 * j;
            xyz[2] = 30.0 * i + 1.0 * j;
            oskar_station_set_element_coords(st, 0, j, xyz, xyz, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
        }
    }

    // Save the telescope model.
    oskar_telescope_save(telescope, tm, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);

    // Load it back again.
    oskar_Telescope* telescope2 = oskar_telescope_create(OSKAR_SINGLE,
                    OSKAR_CPU, 0, &err);
    oskar_telescope_set_position(telescope2,
            longitude_rad, latitude_rad, altitude_m);
    oskar_telescope_set_enable_numerical_patterns(telescope, 0);
    oskar_telescope_load(telescope2, tm, NULL, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);

    // Check the contents.
    ASSERT_EQ(oskar_telescope_num_stations(telescope),
            oskar_telescope_num_stations(telescope2));
    EXPECT_NEAR(oskar_telescope_lon_rad(telescope),
            oskar_telescope_lon_rad(telescope2), 1e-10);
    EXPECT_NEAR(oskar_telescope_lat_rad(telescope),
            oskar_telescope_lat_rad(telescope2), 1e-10);
    EXPECT_NEAR(oskar_telescope_alt_metres(telescope),
            oskar_telescope_alt_metres(telescope2), 1e-10);

    double max_ = 0.0, avg_ = 0.0;
    for (int dim = 0; dim < 3; dim++)
    {
        oskar_mem_evaluate_relative_error(
                oskar_telescope_station_true_enu_metres(telescope, dim),
                oskar_telescope_station_true_enu_metres(telescope2, dim),
                0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);
    }
    for (int i = 0; i < num_stations; ++i)
    {
        oskar_Station* st1 = oskar_telescope_station(telescope, i);
        oskar_Station* st2 = oskar_telescope_station(telescope2, i);
        for (int dim = 0; dim < 3; dim++)
        {
            oskar_mem_evaluate_relative_error(
                    oskar_station_element_measured_enu_metres(st1, 0, dim),
                    oskar_station_element_measured_enu_metres(st2, 0, dim),
                    0, &max_, &avg_, 0, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
            EXPECT_LT(max_, 1e-5);
            EXPECT_LT(avg_, 1e-5);
        }
    }

    // Remove test directory.
    oskar_dir_remove(tm);

    // Free models.
    oskar_telescope_free(telescope, &err);
    oskar_telescope_free(telescope2, &err);
}


TEST(telescope_model_load_save, test_2_level)
{
    int err = 0;
    const char* tm = "temp_test_telescope_2_level";

    // Create a telescope model.
    int num_stations = 3;
    int num_tiles = 4;
    int num_elements = 8;
    double longitude_rad = 0.1;
    double latitude_rad = 0.5;
    double altitude_m = 1.0;
    oskar_Telescope* telescope = oskar_telescope_create(OSKAR_SINGLE,
            OSKAR_CPU, num_stations, &err);
    oskar_telescope_resize_station_array(telescope, num_stations, &err);
    oskar_telescope_set_unique_stations(telescope, 1, &err);
    oskar_telescope_set_position(telescope, longitude_rad,
            latitude_rad, altitude_m);

    // Populate a telescope model.
    for (int i = 0; i < num_stations; ++i)
    {
        double xyz[3];
        xyz[0] = 1.0 * i;
        xyz[1] = 2.0 * i;
        xyz[2] = 3.0 * i;
        oskar_Station* st = oskar_telescope_station(telescope, i);
        oskar_telescope_set_station_coords(telescope, i, xyz,
                xyz, xyz, xyz, xyz, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        oskar_station_resize(st, num_tiles, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        oskar_station_create_child_stations(st, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        for (int j = 0; j < num_tiles; ++j)
        {
            xyz[0] = 10.0 * i + 1.0 * j;
            xyz[1] = 20.0 * i + 1.0 * j;
            xyz[2] = 30.0 * i + 1.0 * j;
            oskar_station_set_element_coords(st, 0, j, xyz, xyz, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
            oskar_station_resize(oskar_station_child(st, j), num_elements, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);

            for (int k = 0; k < num_elements; ++k)
            {
                xyz[0] = 100.0 * i + 10.0 * j + 1.0 * k;
                xyz[1] = 200.0 * i + 10.0 * j + 1.0 * k;
                xyz[2] = 300.0 * i + 10.0 * j + 1.0 * k;
                oskar_station_set_element_coords(oskar_station_child(st, j), 0,
                        k, xyz, xyz, &err);
                ASSERT_EQ(0, err) << oskar_get_error_string(err);
            }
        }
    }

    // Save the telescope model.
    oskar_telescope_save(telescope, tm, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);

    // Load it back again.
    oskar_Telescope* telescope2 = oskar_telescope_create(OSKAR_SINGLE,
                    OSKAR_CPU, 0, &err);
    oskar_telescope_set_position(telescope2,
            longitude_rad, latitude_rad, altitude_m);
    oskar_telescope_set_enable_numerical_patterns(telescope, 0);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    oskar_telescope_load(telescope2, tm, NULL, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);

    // Check the contents.
    ASSERT_EQ(oskar_telescope_num_stations(telescope),
            oskar_telescope_num_stations(telescope2));
    EXPECT_NEAR(oskar_telescope_lon_rad(telescope),
            oskar_telescope_lon_rad(telescope2), 1e-10);
    EXPECT_NEAR(oskar_telescope_lat_rad(telescope),
            oskar_telescope_lat_rad(telescope2), 1e-10);
    EXPECT_NEAR(oskar_telescope_alt_metres(telescope),
            oskar_telescope_alt_metres(telescope2), 1e-10);

    double max_ = 0.0, avg_ = 0.0;
    for (int dim = 0; dim < 3; dim++)
    {
        oskar_mem_evaluate_relative_error(
                oskar_telescope_station_true_enu_metres(telescope, dim),
                oskar_telescope_station_true_enu_metres(telescope2, dim),
                0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);
    }
    for (int i = 0; i < num_stations; ++i)
    {
        oskar_Station* s1 = oskar_telescope_station(telescope, i);
        oskar_Station* s2 = oskar_telescope_station(telescope2, i);
        for (int dim = 0; dim < 3; dim++)
        {
            oskar_mem_evaluate_relative_error(
                    oskar_station_element_measured_enu_metres(s1, 0, dim),
                    oskar_station_element_measured_enu_metres(s2, 0, dim),
                    0, &max_, &avg_, 0, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
            EXPECT_LT(max_, 1e-5);
            EXPECT_LT(avg_, 1e-5);
        }

        for (int j = 0; j < num_tiles; ++j)
        {
            oskar_Station *c1 = oskar_station_child(s1, j);
            oskar_Station *c2 = oskar_station_child(s2, j);
            for (int dim = 0; dim < 3; dim++)
            {
                oskar_mem_evaluate_relative_error(
                        oskar_station_element_measured_enu_metres(c1, 0, dim),
                        oskar_station_element_measured_enu_metres(c2, 0, dim),
                        0, &max_, &avg_, 0, &err);
                ASSERT_EQ(0, err) << oskar_get_error_string(err);
                EXPECT_LT(max_, 1e-5);
                EXPECT_LT(avg_, 1e-5);
            }
        }
    }

    // Copy the telescope model to a new structure.
    oskar_Telescope* telescope3 = oskar_telescope_create_copy(telescope2,
            OSKAR_CPU, &err);

    // Check the contents.
    ASSERT_EQ(oskar_telescope_num_stations(telescope),
            oskar_telescope_num_stations(telescope3));
    EXPECT_NEAR(oskar_telescope_lon_rad(telescope),
            oskar_telescope_lon_rad(telescope3), 1e-10);
    EXPECT_NEAR(oskar_telescope_lat_rad(telescope),
            oskar_telescope_lat_rad(telescope3), 1e-10);
    EXPECT_NEAR(oskar_telescope_alt_metres(telescope),
            oskar_telescope_alt_metres(telescope3), 1e-10);

    for (int dim = 0; dim < 3; dim++)
    {
        oskar_mem_evaluate_relative_error(
                oskar_telescope_station_true_enu_metres(telescope, dim),
                oskar_telescope_station_true_enu_metres(telescope3, dim),
                0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);
    }
    for (int i = 0; i < num_stations; ++i)
    {
        oskar_Station* s1 = oskar_telescope_station(telescope, i);
        oskar_Station* s3 = oskar_telescope_station(telescope3, i);
        for (int dim = 0; dim < 3; dim++)
        {
            oskar_mem_evaluate_relative_error(
                    oskar_station_element_measured_enu_metres(s1, 0, dim),
                    oskar_station_element_measured_enu_metres(s3, 0, dim),
                    0, &max_, &avg_, 0, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
            EXPECT_LT(max_, 1e-5);
            EXPECT_LT(avg_, 1e-5);
        }

        for (int j = 0; j < num_tiles; ++j)
        {
            oskar_Station *c1 = oskar_station_child(s1, j);
            oskar_Station *c3 = oskar_station_child(s3, j);
            for (int dim = 0; dim < 3; dim++)
            {
                oskar_mem_evaluate_relative_error(
                        oskar_station_element_measured_enu_metres(c1, 0, dim),
                        oskar_station_element_measured_enu_metres(c3, 0, dim),
                        0, &max_, &avg_, 0, &err);
                ASSERT_EQ(0, err) << oskar_get_error_string(err);
                EXPECT_LT(max_, 1e-5);
                EXPECT_LT(avg_, 1e-5);
            }
        }
    }

    // Free models.
    oskar_telescope_free(telescope, &err);
    oskar_telescope_free(telescope2, &err);
    oskar_telescope_free(telescope3, &err);

    // Remove test directory.
    oskar_dir_remove(tm);
}

//
// TODO: check combinations of telescope model loading and overrides...
//

TEST(telescope_model_load_save, test_load_telescope_noise_rms)
{
    // Test cases that should be considered.
    // -- stddev file at various depths
    // -- number of noise values vs number of stddev
    // -- different modes of getting stddev and freq data.

    const char* root = "temp_test_noise_rms";
    int err = 0;
    int num_stations = 2;
    int num_values = 5;
    int type = OSKAR_DOUBLE;

    // Generate the telescope model.
    vector<double> stddev(num_values), freq_values(num_values);
    for (int i = 0; i < num_values; ++i)
    {
        stddev[i] = i * 0.25 + 0.5;
        freq_values[i] = 20.0e6 + i * 10.0e6;
    }
    generate_noisy_telescope(root, num_stations, freq_values, stddev);

    // Load it back again.
    oskar_Telescope* telescope = oskar_telescope_create(type,
            OSKAR_CPU, 0, &err);
    oskar_telescope_set_enable_numerical_patterns(telescope, 0);
    oskar_telescope_set_enable_noise(telescope, true, 1);
    oskar_telescope_load(telescope, root, NULL, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    ASSERT_EQ(num_stations, oskar_telescope_num_stations(telescope));

    // Check the loaded std.dev. values
    for (int i = 0; i < oskar_telescope_num_station_models(telescope); ++i)
    {
        oskar_Station* s = oskar_telescope_station(telescope, i);
        oskar_Mem *freq = 0, *rms = 0;
        freq = oskar_station_noise_freq_hz(s);
        rms = oskar_station_noise_rms_jy(s);
        ASSERT_EQ(num_values, (int)oskar_mem_length(rms));
        ASSERT_EQ(num_values, (int)oskar_mem_length(freq));
        if (type == OSKAR_DOUBLE)
        {
            double* r = oskar_mem_double(rms, &err);
            double* f = oskar_mem_double(freq, &err);
            for (int j = 0; j < num_values; ++j)
            {
                EXPECT_NEAR(stddev[j], r[j], 1.0e-6);
                EXPECT_NEAR(freq_values[j], f[j], 1.0e-6);
            }
        }
        else
        {
            float* r = oskar_mem_float(rms, &err);
            float* f = oskar_mem_float(freq, &err);
            for (int j = 0; j < num_values; ++j)
            {
                EXPECT_NEAR(stddev[j], r[j], 1.0e-5);
                EXPECT_NEAR(freq_values[j], f[j], 1.0e-5);
            }
        }
    }

    oskar_dir_remove(root);
    oskar_telescope_free(telescope, &err);
}


static void generate_noisy_telescope(const char* dir, int num_stations,
        const vector<double>& freqs, const vector<double>& noise)
{
    FILE* f = 0;
    char* path = 0;

    // Create a telescope model directory.
    if (oskar_dir_exists(dir)) oskar_dir_remove(dir);
    oskar_dir_mkpath(dir);

    // Write position file.
    path = oskar_dir_get_path(dir, "position.txt");
    f = fopen(path, "w");
    fprintf(f, "0,0\n");
    fclose(f);
    free(path);

    // Write the layout file.
    path = oskar_dir_get_path(dir, "layout.txt");
    f = fopen(path, "w");
    for (int i = 0; i < num_stations; ++i) fprintf(f, "0,0\n");
    fclose(f);
    free(path);

    // Write frequency file.
    if (!freqs.empty())
    {
        path = oskar_dir_get_path(dir, "noise_frequencies.txt");
        f = fopen(path, "w");
        for (size_t i = 0; i < freqs.size(); ++i)
        {
            fprintf(f, "%.10f\n", freqs[i]);
        }
        fclose(f);
        free(path);
    }

    // Write RMS noise values.
    if (!noise.empty())
    {
        path = oskar_dir_get_path(dir, "rms.txt");
        f = fopen(path, "w");
        for (size_t i = 0; i < noise.size(); ++i)
        {
            fprintf(f, "%.10f\n", noise[i]);
        }
        fclose(f);
        free(path);
    }
}
