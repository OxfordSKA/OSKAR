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

#include "apps/lib/oskar_remove_dir.h"
#include "apps/lib/oskar_telescope_save.h"
#include "apps/lib/oskar_telescope_load.h"

#include <oskar_SettingsTelescope.h>
#include <oskar_telescope.h>
#include <oskar_get_error_string.h>
#include <oskar_settings_init.h>
#include <oskar_settings_free.h>
#include <oskar_mem.h>

#include <QtCore/QtCore>

#include <cstdio>
#include <cstdlib>

static void generate_noisy_telescope(const QString& dir,
        int num_stations, int write_depth, const QVector<double>& freqs,
        const QHash< QString, QVector<double> >& noise);

TEST(telescope_model_load_save, test_0_level)
{
    int err = 0;
    const char* path = "temp_test_telescope_0_level";
    double longitude_rad = 0.1;
    double latitude_rad = 0.5;
    double altitude_m = 1.0;

    {
        int num_stations = 10;

        // Create a telescope model directory.
        QDir cwd;
        cwd.mkdir(path);
        cwd.cd(path);

        // Write the top-level layout file only.
        QFile file(cwd.absoluteFilePath("layout.txt"));
        file.open(QFile::WriteOnly);
        QTextStream stream(&file);
        for (int i = 0; i < num_stations; ++i)
        {
            stream << i * 10.0 << "," << i * 20.0 << "," << i * 30.0 << endl;
        }
    }

    // Load it back again.
    {
        oskar_Settings settings;
        oskar_settings_init(&settings);
        settings.telescope.altitude_m    = altitude_m;
        settings.telescope.latitude_rad  = latitude_rad;
        settings.telescope.longitude_rad = longitude_rad;
        settings.telescope.input_directory = (char*)malloc(1 + strlen(path));
        settings.telescope.aperture_array.element_pattern.enable_numerical_patterns = false;
        settings.interferometer.noise.enable = false;
        strcpy(settings.telescope.input_directory, path);

        oskar_Telescope* telescope = oskar_telescope_create(OSKAR_DOUBLE,
                        OSKAR_LOCATION_CPU, 0, &err);
        oskar_telescope_load(telescope, NULL, &settings, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        oskar_telescope_free(telescope, &err);
        oskar_settings_free(&settings);
    }

    // Remove test directory.
    oskar_remove_dir(path);
}


TEST(telescope_model_load_save, test_1_level)
{
    int err = 0;
    const char* path = "temp_test_telescope_1_level";

    // Create a telescope model.
    int num_stations = 10;
    int num_elements = 20;
    double longitude_rad = 0.1;
    double latitude_rad = 0.5;
    double altitude_m = 1.0;
    oskar_Telescope* telescope = oskar_telescope_create(OSKAR_SINGLE,
            OSKAR_LOCATION_CPU, num_stations, &err);
    oskar_telescope_set_position(telescope, longitude_rad,
            latitude_rad, altitude_m);

    // Populate a telescope model.
    for (int i = 0; i < num_stations; ++i)
    {
        oskar_Station* st = oskar_telescope_station(telescope, i);
        oskar_telescope_set_station_coords(telescope, i, 0.0, 0.0, 0.0,
                (double) i, (double) (2 * i), (double) (3 * i), &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        oskar_station_resize(st, num_elements, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        for (int j = 0; j < num_elements; ++j)
        {
            oskar_station_set_element_coords(st,
                    j, (double) (10 * i + j), (double) (20 * i + j),
                    (double) (30 * i + j), 0.0, 0.0, 0.0, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
        }
    }

    // Save the telescope model.
    oskar_telescope_save(telescope, path, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);

    // Load it back again.
    oskar_Settings settings;
    oskar_settings_init(&settings);
    settings.telescope.altitude_m = altitude_m;
    settings.telescope.latitude_rad = latitude_rad;
    settings.telescope.longitude_rad = longitude_rad;
    settings.telescope.input_directory = (char*)malloc(1 + strlen(path));
    settings.telescope.aperture_array.element_pattern.enable_numerical_patterns = false;
    settings.interferometer.noise.enable = false;
    strcpy(settings.telescope.input_directory, path);
    oskar_Telescope* telescope2 =
            oskar_telescope_create(OSKAR_SINGLE,
                    OSKAR_LOCATION_CPU, 0, &err);
    oskar_telescope_load(telescope2, NULL, &settings, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    oskar_settings_free(&settings);

    // Check the contents.
    ASSERT_EQ(oskar_telescope_num_stations(telescope),
            oskar_telescope_num_stations(telescope2));
    EXPECT_NEAR(oskar_telescope_longitude_rad(telescope),
            oskar_telescope_longitude_rad(telescope2), 1e-10);
    EXPECT_NEAR(oskar_telescope_latitude_rad(telescope),
            oskar_telescope_latitude_rad(telescope2), 1e-10);
    EXPECT_NEAR(oskar_telescope_altitude_m(telescope),
            oskar_telescope_altitude_m(telescope2), 1e-10);

    double max_, avg_;
    oskar_mem_evaluate_relative_error(
            oskar_telescope_station_x_hor(telescope),
            oskar_telescope_station_x_hor(telescope2),
            0, &max_, &avg_, 0, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    EXPECT_LT(max_, 1e-5);
    EXPECT_LT(avg_, 1e-5);
    oskar_mem_evaluate_relative_error(
            oskar_telescope_station_y_hor(telescope),
            oskar_telescope_station_y_hor(telescope2),
            0, &max_, &avg_, 0, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    EXPECT_LT(max_, 1e-5);
    EXPECT_LT(avg_, 1e-5);
    oskar_mem_evaluate_relative_error(
            oskar_telescope_station_z_hor(telescope),
            oskar_telescope_station_z_hor(telescope2),
            0, &max_, &avg_, 0, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    EXPECT_LT(max_, 1e-5);
    EXPECT_LT(avg_, 1e-5);
    for (int i = 0; i < num_stations; ++i)
    {
        oskar_Station* st1 = oskar_telescope_station(telescope, i);
        oskar_Station* st2 = oskar_telescope_station(telescope2, i);
        oskar_mem_evaluate_relative_error(oskar_station_element_x_weights(st1),
                oskar_station_element_x_weights(st2), 0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);
        oskar_mem_evaluate_relative_error(oskar_station_element_y_weights(st1),
                oskar_station_element_y_weights(st2), 0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);
        oskar_mem_evaluate_relative_error(oskar_station_element_z_weights(st1),
                oskar_station_element_z_weights(st2), 0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);
    }

    // Remove test directory.
    oskar_remove_dir(path);

    // Free models.
    oskar_telescope_free(telescope, &err);
    oskar_telescope_free(telescope2, &err);
}


TEST(telescope_model_load_save, test_2_level)
{
    int err = 0;
    const char* path = "temp_test_telescope_2_level";

    // Create a telescope model.
    int num_stations = 3;
    int num_tiles = 4;
    int num_elements = 8;
    double longitude_rad = 0.1;
    double latitude_rad = 0.5;
    double altitude_m = 1.0;
    oskar_Telescope* telescope = oskar_telescope_create(OSKAR_SINGLE,
            OSKAR_LOCATION_CPU, num_stations, &err);
    oskar_telescope_set_position(telescope, longitude_rad,
            latitude_rad, altitude_m);

    // Populate a telescope model.
    for (int i = 0; i < num_stations; ++i)
    {
        oskar_Station* st = oskar_telescope_station(telescope, i);
        oskar_telescope_set_station_coords(telescope, i, 0.0, 0.0, 0.0,
                (double) i, (double) (2 * i), (double) (3 * i), &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        oskar_station_resize(st, num_tiles, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        oskar_station_create_child_stations(st, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        for (int j = 0; j < num_tiles; ++j)
        {
            oskar_station_set_element_coords(st,
                    j, (double) (10 * i + j), (double) (20 * i + j),
                    (double) (30 * i + j), 0.0, 0.0, 0.0, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
            oskar_station_resize(oskar_station_child(st, j), num_elements, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);

            for (int k = 0; k < num_elements; ++k)
            {
                oskar_station_set_element_coords(oskar_station_child(st, j),
                        k, (double) (100 * i + 10 * j + k),
                        (double) (200 * i + 20 * j + k),
                        (double) (300 * i + 30 * j + k), 0.0, 0.0, 0.0, &err);
                ASSERT_EQ(0, err) << oskar_get_error_string(err);
            }
        }
    }

    // Save the telescope model.
    oskar_telescope_save(telescope, path, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);

    // Add an element pattern file.
    QFile file(path + QString("/level0_000/element_pattern_x.txt"));
    file.open(QFile::WriteOnly);
    file.write("\n");
    file.close();

    // Load it back again.
    oskar_Settings settings;
    oskar_settings_init(&settings);
    settings.telescope.altitude_m = altitude_m;
    settings.telescope.latitude_rad = latitude_rad;
    settings.telescope.longitude_rad = longitude_rad;
    settings.telescope.input_directory = (char*)malloc(1 + strlen(path));
    settings.telescope.aperture_array.element_pattern.enable_numerical_patterns = false;
    settings.interferometer.noise.enable = false;
    strcpy(settings.telescope.input_directory, path);
    oskar_Telescope* telescope2 =
            oskar_telescope_create(OSKAR_SINGLE,
                    OSKAR_LOCATION_CPU, 0, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    oskar_telescope_load(telescope2, NULL, &settings, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    oskar_settings_free(&settings);

    // Check the contents.
    ASSERT_EQ(oskar_telescope_num_stations(telescope),
            oskar_telescope_num_stations(telescope2));
    EXPECT_NEAR(oskar_telescope_longitude_rad(telescope),
            oskar_telescope_longitude_rad(telescope2), 1e-10);
    EXPECT_NEAR(oskar_telescope_latitude_rad(telescope),
            oskar_telescope_latitude_rad(telescope2), 1e-10);
    EXPECT_NEAR(oskar_telescope_altitude_m(telescope),
            oskar_telescope_altitude_m(telescope2), 1e-10);

    double max_, avg_ = 0.0;
    oskar_mem_evaluate_relative_error(
            oskar_telescope_station_x_hor(telescope),
            oskar_telescope_station_x_hor(telescope2),
            0, &max_, &avg_, 0, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    EXPECT_LT(max_, 1e-5);
    EXPECT_LT(avg_, 1e-5);
    oskar_mem_evaluate_relative_error(
            oskar_telescope_station_y_hor(telescope),
            oskar_telescope_station_y_hor(telescope2),
            0, &max_, &avg_, 0, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    EXPECT_LT(max_, 1e-5);
    EXPECT_LT(avg_, 1e-5);
    oskar_mem_evaluate_relative_error(
            oskar_telescope_station_z_hor(telescope),
            oskar_telescope_station_z_hor(telescope2),
            0, &max_, &avg_, 0, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    EXPECT_LT(max_, 1e-5);
    EXPECT_LT(avg_, 1e-5);
    for (int i = 0; i < num_stations; ++i)
    {
        oskar_Station* s1 = oskar_telescope_station(telescope, i);
        oskar_Station* s2 = oskar_telescope_station(telescope2, i);
        oskar_mem_evaluate_relative_error(oskar_station_element_x_weights(s1),
                oskar_station_element_x_weights(s2), 0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);
        oskar_mem_evaluate_relative_error(oskar_station_element_y_weights(s1),
                oskar_station_element_y_weights(s2), 0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);
        oskar_mem_evaluate_relative_error(oskar_station_element_z_weights(s1),
                oskar_station_element_z_weights(s2), 0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);

        for (int j = 0; j < num_tiles; ++j)
        {
            oskar_Station *c1 = oskar_station_child(s1, j);
            oskar_Station *c2 = oskar_station_child(s2, j);
            oskar_mem_evaluate_relative_error(
                    oskar_station_element_x_weights(c1),
                    oskar_station_element_x_weights(c2),
                    0, &max_, &avg_, 0, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
            EXPECT_LT(max_, 1e-5);
            EXPECT_LT(avg_, 1e-5);
            oskar_mem_evaluate_relative_error(
                    oskar_station_element_y_weights(c1),
                    oskar_station_element_y_weights(c2),
                    0, &max_, &avg_, 0, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
            EXPECT_LT(max_, 1e-5);
            EXPECT_LT(avg_, 1e-5);
            oskar_mem_evaluate_relative_error(
                    oskar_station_element_z_weights(c1),
                    oskar_station_element_z_weights(c2),
                    0, &max_, &avg_, 0, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
            EXPECT_LT(max_, 1e-5);
            EXPECT_LT(avg_, 1e-5);
        }
    }

    // Copy the telescope model to a new structure.
    oskar_Telescope* telescope3 = oskar_telescope_create_copy(telescope2,
            OSKAR_LOCATION_CPU, &err);

    // Check the contents.
    ASSERT_EQ(oskar_telescope_num_stations(telescope),
            oskar_telescope_num_stations(telescope3));
    EXPECT_NEAR(oskar_telescope_longitude_rad(telescope),
            oskar_telescope_longitude_rad(telescope3), 1e-10);
    EXPECT_NEAR(oskar_telescope_latitude_rad(telescope),
            oskar_telescope_latitude_rad(telescope3), 1e-10);
    EXPECT_NEAR(oskar_telescope_altitude_m(telescope),
            oskar_telescope_altitude_m(telescope3), 1e-10);

    oskar_mem_evaluate_relative_error(
            oskar_telescope_station_x_hor(telescope),
            oskar_telescope_station_x_hor(telescope3),
            0, &max_, &avg_, 0, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    EXPECT_LT(max_, 1e-5);
    EXPECT_LT(avg_, 1e-5);
    oskar_mem_evaluate_relative_error(
            oskar_telescope_station_y_hor(telescope),
            oskar_telescope_station_y_hor(telescope3),
            0, &max_, &avg_, 0, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    EXPECT_LT(max_, 1e-5);
    EXPECT_LT(avg_, 1e-5);
    oskar_mem_evaluate_relative_error(
            oskar_telescope_station_z_hor(telescope),
            oskar_telescope_station_z_hor(telescope3),
            0, &max_, &avg_, 0, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    EXPECT_LT(max_, 1e-5);
    EXPECT_LT(avg_, 1e-5);
    for (int i = 0; i < num_stations; ++i)
    {
        oskar_Station* s1 = oskar_telescope_station(telescope, i);
        oskar_Station* s3 = oskar_telescope_station(telescope3, i);
        oskar_mem_evaluate_relative_error(oskar_station_element_x_weights(s1),
                oskar_station_element_x_weights(s3), 0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);
        oskar_mem_evaluate_relative_error(oskar_station_element_y_weights(s1),
                oskar_station_element_y_weights(s3), 0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);
        oskar_mem_evaluate_relative_error(oskar_station_element_z_weights(s1),
                oskar_station_element_z_weights(s3), 0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);

        for (int j = 0; j < num_tiles; ++j)
        {
            oskar_Station *c1 = oskar_station_child(s1, j);
            oskar_Station *c3 = oskar_station_child(s3, j);
            oskar_mem_evaluate_relative_error(
                    oskar_station_element_x_weights(c1),
                    oskar_station_element_x_weights(c3),
                    0, &max_, &avg_, 0, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
            EXPECT_LT(max_, 1e-5);
            EXPECT_LT(avg_, 1e-5);
            oskar_mem_evaluate_relative_error(
                    oskar_station_element_y_weights(c1),
                    oskar_station_element_y_weights(c3),
                    0, &max_, &avg_, 0, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
            EXPECT_LT(max_, 1e-5);
            EXPECT_LT(avg_, 1e-5);
            oskar_mem_evaluate_relative_error(
                    oskar_station_element_z_weights(c1),
                    oskar_station_element_z_weights(c3),
                    0, &max_, &avg_, 0, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
            EXPECT_LT(max_, 1e-5);
            EXPECT_LT(avg_, 1e-5);
        }
    }

    // Free models.
    oskar_telescope_free(telescope, &err);
    oskar_telescope_free(telescope2, &err);
    oskar_telescope_free(telescope3, &err);

    // Remove test directory.
    oskar_remove_dir(path);
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

    QString root = "./temp_test_noise_rms";
    int err = 0;
    int num_stations = 2;
    int depth = 0;
    int num_values = 5;
    int type = OSKAR_DOUBLE;
    int location = OSKAR_LOCATION_CPU;

    QVector<double> stddev(num_values);
    for (int i = 0; i < num_values; ++i)
    {
        stddev[i] = i * 0.25 + 0.5;
    }

    QVector<double> freq_values(num_values);
    for (int i = 0; i < num_values; ++i)
    {
        freq_values[i] = 20.0e6 + i * 10.0e6;
    }

    QHash<QString, QVector<double> > noise_;
    noise_["rms.txt"] = stddev;

    // Generate the telescope
    generate_noisy_telescope(root, num_stations, depth, freq_values, noise_);

    oskar_Telescope* telescope = oskar_telescope_create(type,
            location, 0, &err);
    oskar_Settings settings;
    oskar_settings_init(&settings);
    settings.telescope.altitude_m = 0.0;
    settings.telescope.latitude_rad = 0.5;
    settings.telescope.longitude_rad = 0.1;
    settings.sim.double_precision = (type == OSKAR_DOUBLE) ? OSKAR_TRUE : OSKAR_FALSE;
    QByteArray path = root.toLatin1();
    settings.telescope.input_directory = (char*)malloc(root.size() + 1);
    strcpy(settings.telescope.input_directory, path.constData());
    settings.telescope.aperture_array.element_pattern.enable_numerical_patterns = false;
    oskar_SettingsSystemNoise* noise = &settings.interferometer.noise;
    noise->enable = OSKAR_TRUE;
    noise->seed = 0;
    noise->value.specification = OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL;
    noise->freq.specification = OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL;
    settings.obs.length_seconds = 1;
    settings.obs.num_time_steps = 1;
    settings.interferometer.channel_bandwidth_hz = 1;
    settings.interferometer.time_average_sec = 1;

    oskar_telescope_load(telescope, NULL, &settings, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);

    ASSERT_EQ(oskar_telescope_num_stations(telescope), num_stations);
    // Check the loaded std.dev. values
    for (int i = 0; i < num_stations; ++i)
    {
        oskar_Station* s = oskar_telescope_station(telescope, i);
        oskar_SystemNoiseModel* noise = oskar_station_system_noise_model(s);
        oskar_Mem *freq, *rms;
        freq = oskar_system_noise_model_frequency(noise);
        rms = oskar_system_noise_model_rms(noise);
        int num_values = (int)oskar_mem_length(freq);
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

    oskar_remove_dir(path.data());
    oskar_telescope_free(telescope, &err);
}


TEST(telescope_model_load_save, test_load_telescope_noise_sensitivity)
{
    QString root = "./temp_test_noise_sensitivity";
    int err = 0;
    int num_stations = 2;
    int depth = 1;
    int num_values = 5;
    int type = OSKAR_DOUBLE;
    int location = OSKAR_LOCATION_CPU;

    QVector<double> sensitivity(num_values);
    for (int i = 0; i < num_values; ++i)
    {
        sensitivity[i] = i * 0.25 + 0.5;
    }

    QVector<double> freq_values(num_values);
    for (int i = 0; i < num_values; ++i)
    {
        freq_values[i] = 20.0e6 + i * 10.0e6;
    }

    QHash<QString, QVector<double> > noise_;
    noise_["sensitivity.txt"] = sensitivity;

    // Generate the telescope
    generate_noisy_telescope(root, num_stations, depth, freq_values, noise_);

    oskar_Telescope* telescope = oskar_telescope_create(type,
            location, 0, &err);
    oskar_Settings settings;
    oskar_settings_init(&settings);
    settings.telescope.altitude_m = 0.0;
    settings.telescope.latitude_rad = 0.5;
    settings.telescope.longitude_rad = 0.1;
    settings.sim.double_precision = (type == OSKAR_DOUBLE) ? OSKAR_TRUE : OSKAR_FALSE;
    QByteArray path = root.toLatin1();
    settings.telescope.input_directory = (char*)malloc(root.size() + 1);
    strcpy(settings.telescope.input_directory, path.constData());
    settings.telescope.aperture_array.element_pattern.enable_numerical_patterns = false;
    oskar_SettingsSystemNoise* noise = &settings.interferometer.noise;
    noise->enable = OSKAR_TRUE;
    noise->seed = 0;
    noise->value.specification = OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL;
    noise->freq.specification = OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL;
    settings.obs.length_seconds = 1;
    settings.obs.num_time_steps = 1;

    double bandwidth = 10.0e6;
    settings.interferometer.channel_bandwidth_hz = bandwidth;
    settings.obs.length_seconds = 60 * 60;
    settings.obs.num_time_steps = 36;
    double integration_time = settings.obs.length_seconds /
            (double)settings.obs.num_time_steps;
    settings.interferometer.time_average_sec = integration_time;

    oskar_telescope_load(telescope, NULL, &settings, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);

    ASSERT_EQ(oskar_telescope_num_stations(telescope), num_stations);
    // Check the loaded std.dev. values
    for (int i = 0; i < num_stations; ++i)
    {
        oskar_Station* s = oskar_telescope_station(telescope, i);
        oskar_SystemNoiseModel* noise = oskar_station_system_noise_model(s);
        oskar_Mem *freq, *rms;
        freq = oskar_system_noise_model_frequency(noise);
        rms = oskar_system_noise_model_rms(noise);
        int num_values = (int)oskar_mem_length(freq);
        if (type == OSKAR_DOUBLE)
        {
            double* r = oskar_mem_double(rms, &err);
            double* f = oskar_mem_double(freq, &err);
            for (int j = 0; j < num_values; ++j)
            {
                EXPECT_NEAR(
                        sensitivity[j] / sqrt(2.0 * bandwidth * integration_time),
                        r[j], 1.0e-6);
                EXPECT_NEAR(freq_values[j], f[j], 1.0e-6);
            }
        }
        else
        {
            float* r = oskar_mem_float(rms, &err);
            float* f = oskar_mem_float(freq, &err);
            for (int j = 0; j < num_values; ++j)
            {
                EXPECT_NEAR(
                        sensitivity[j] / sqrtf(2.0 * bandwidth * integration_time),
                        r[j], 1.0e-5);
                EXPECT_NEAR(freq_values[j], f[j], 1.0e-5);
            }
        }
    }

    oskar_remove_dir(path.data());
    oskar_telescope_free(telescope, &err);
}


TEST(telescope_model_load_save, test_load_telescope_noise_t_sys)
{
    QString root = "./temp_test_noise_t_sys";
    int err = 0;
    int num_stations = 2;
    int depth = 1;
    int num_values = 5;
    int type = OSKAR_DOUBLE;
    int location = OSKAR_LOCATION_CPU;

    QVector<double> t_sys(num_values);
    for (int i = 0; i < num_values; ++i)
    {
        t_sys[i] = i * 0.25 + 0.5;
    }

    QVector<double> freq_values(num_values);
    for (int i = 0; i < num_values; ++i)
    {
        freq_values[i] = 20.0e6 + i * 10.0e6;
    }

    QVector<double> area_values(num_values);
    for (int i = 0; i < num_values; ++i)
    {
        area_values[i] = i * 5.3 + 1000.0;
    }

    QVector<double> efficiency_values(num_values);
    for (int i = 0; i < num_values; ++i)
    {
        efficiency_values[i] = 0.8;
    }

    QHash<QString, QVector<double> > noise_;
    noise_["t_sys.txt"] = t_sys;
    noise_["area.txt"] = area_values;
    noise_["efficiency.txt"] = efficiency_values;

    // Generate the telescope
    generate_noisy_telescope(root, num_stations, depth, freq_values, noise_);

    oskar_Telescope* telescope = oskar_telescope_create(type,
            location, 0, &err);
    oskar_Settings settings;
    oskar_settings_init(&settings);
    settings.telescope.altitude_m = 0.0;
    settings.telescope.latitude_rad = 0.5;
    settings.telescope.longitude_rad = 0.1;
    settings.sim.double_precision = (type == OSKAR_DOUBLE) ? OSKAR_TRUE : OSKAR_FALSE;
    QByteArray path = root.toLatin1();
    settings.telescope.input_directory = (char*)malloc(root.size() + 1);
    strcpy(settings.telescope.input_directory, path.constData());
    settings.telescope.aperture_array.element_pattern.enable_numerical_patterns = false;
    oskar_SettingsSystemNoise* noise = &settings.interferometer.noise;
    noise->enable = OSKAR_TRUE;
    noise->seed = 0;
    noise->value.specification = OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL;
    noise->freq.specification = OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL;
    settings.obs.length_seconds = 1;
    settings.obs.num_time_steps = 1;

    double bandwidth = 10.0e6;
    settings.interferometer.channel_bandwidth_hz = bandwidth;
    settings.obs.length_seconds = 60 * 60;
    settings.obs.num_time_steps = 36;
    double integration_time = settings.obs.length_seconds /
            (double)settings.obs.num_time_steps;
    settings.interferometer.time_average_sec = integration_time;

    oskar_telescope_load(telescope, NULL, &settings, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);

    ASSERT_EQ(oskar_telescope_num_stations(telescope), num_stations);
    double k_B = 1.3806488e-23;
    double factor = (2.0 * k_B * 1.0e26) / sqrt(2.0 * bandwidth * integration_time);
    // Check the loaded std.dev. values
    for (int i = 0; i < num_stations; ++i)
    {
        oskar_Station* s = oskar_telescope_station(telescope, i);
        oskar_SystemNoiseModel* noise = oskar_station_system_noise_model(s);
        oskar_Mem *freq, *rms;
        freq = oskar_system_noise_model_frequency(noise);
        rms = oskar_system_noise_model_rms(noise);
        int num_values = (int)oskar_mem_length(freq);

        if (type == OSKAR_DOUBLE)
        {
            double* r = oskar_mem_double(rms, &err);
            double* f = oskar_mem_double(freq, &err);
            for (int j = 0; j < num_values; ++j)
            {
                EXPECT_NEAR(
                        (t_sys[j] / (area_values[j] * efficiency_values[j])) * factor,
                        r[j], 1.0e-6);
                EXPECT_NEAR(freq_values[j], f[j], 1.0e-6);
            }
        }
        else
        {
            float* r = oskar_mem_float(rms, &err);
            float* f = oskar_mem_float(freq, &err);
            for (int j = 0; j < num_values; ++j)
            {
                EXPECT_NEAR(
                        (t_sys[j] / (area_values[j] * efficiency_values[j])) * factor,
                        r[j], 1.0e-5);
                EXPECT_NEAR(freq_values[j], f[j], 1.0e-5);
            }
        }
    }

    oskar_remove_dir(path.data());
    oskar_telescope_free(telescope, &err);
}


static void generate_noisy_telescope(const QString& dir,
        int num_stations, int write_depth, const QVector<double>& freqs,
        const QHash< QString, QVector<double> >& noise)
{
    QDir root(dir);

    if (root.exists())
    {
        QByteArray name_ = dir.toLatin1();
        oskar_remove_dir(name_.data());
    }

    root.mkdir(root.absolutePath());

    // Write top-level config file.
    {
        QString config_file = "config.txt";
        QFile file(dir + QDir::separator() + config_file);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
            return;
        QTextStream out(&file);
        for (int i = 0; i < num_stations; ++i)
            out << "0,0" << endl;
    }

    // Write frequency file.
    if (!freqs.isEmpty())
    {
        QString freq_file = "noise_frequencies.txt";
        QFile file(dir + QDir::separator() + freq_file);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
            return;
        QTextStream out(&file);
        out.setRealNumberPrecision(10);
        for (int i = 0; i < freqs.size(); ++i)
            out << freqs[i] << endl;
    }

    if (write_depth == 0)
    {
        QHash<QString, QVector<double> >::const_iterator noise_ = noise.constBegin();
        while (noise_ != noise.constEnd())
        {
            QFile file(dir +  QDir::separator() + noise_.key());
            if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
                return;
            QTextStream out(&file);
            out.setRealNumberPrecision(10);
            for (int i = 0; i < noise_.value().size(); ++i)
                out << noise_.value()[i] << endl;
            ++noise_;
        }
    }


    for (int i = 0; i < num_stations; ++i)
    {
        char name[200];
        sprintf(name, "station%03i", i);
        QString station_name = dir + QDir::separator() + QString(name);
        QDir station_dir(station_name);
        station_dir.mkdir(station_dir.absolutePath());

        // Write station config file.
        {
            QString config_file = "config.txt";
            QFile file(station_name + QDir::separator() + config_file);
            if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
                return;
            QTextStream out(&file);
            out << "0,0" << endl;
        }

        if (write_depth == 1)
        {
            for (int i = 0; i < num_stations; ++i)
            {
                QHash<QString, QVector<double> >::const_iterator noise_ = noise.constBegin();
                while (noise_ != noise.constEnd())
                {
                    QFile file(dir +  QDir::separator() + noise_.key());
                    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
                        return;
                    QTextStream out(&file);
                    out.setRealNumberPrecision(10);
                    for (int i = 0; i < noise_.value().size(); ++i)
                        out << noise_.value()[i] << endl;
                    ++noise_;
                }
            }
        }
    }
}

