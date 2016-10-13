/*
 * Copyright (c) 2012-2016, The University of Oxford
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

#include "apps/lib/oskar_dir.h"
#include "apps/lib/oskar_telescope_save.h"
#include "apps/lib/oskar_telescope_load.h"

#include <oskar_telescope.h>
#include <oskar_get_error_string.h>
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
        FILE* f;
        int num_stations = 10;

        // Create a telescope model directory.
        QDir cwd;
        cwd.mkdir(path);
        cwd.cd(path);

        // Write the top-level layout file only.
        f = fopen(cwd.absoluteFilePath("layout.txt").toLatin1().data(), "w");
        for (int i = 0; i < num_stations; ++i)
            fprintf(f, "%.1f, %.1f, %.1f\n", i * 10.0, i * 20.0, i * 30.0);
        fclose(f);
        f = fopen(cwd.absoluteFilePath("position.txt").toLatin1().data(), "w");
        fprintf(f, "0.0, 0.0\n");
        fclose(f);
    }

    // Load it back again.
    {
        oskar_Telescope* telescope = oskar_telescope_create(OSKAR_DOUBLE,
                        OSKAR_CPU, 0, &err);
        oskar_telescope_set_position(telescope,
                longitude_rad, latitude_rad, altitude_m);
        oskar_telescope_set_enable_numerical_patterns(telescope, 0);
        oskar_telescope_load(telescope, path, NULL, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        oskar_telescope_free(telescope, &err);
    }

    // Remove test directory.
    oskar_dir_remove(path);
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
            OSKAR_CPU, num_stations, &err);
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
        oskar_telescope_set_station_coords(telescope, i,
                xyz, xyz, xyz, xyz, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        oskar_station_resize(st, num_elements, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        for (int j = 0; j < num_elements; ++j)
        {
            xyz[0] = 10.0 * i + 1.0 * j;
            xyz[1] = 20.0 * i + 1.0 * j;
            xyz[2] = 30.0 * i + 1.0 * j;
            oskar_station_set_element_coords(st, j, xyz, xyz, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
        }
    }

    // Save the telescope model.
    oskar_telescope_save(telescope, path, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);

    // Load it back again.
    oskar_Telescope* telescope2 = oskar_telescope_create(OSKAR_SINGLE,
                    OSKAR_CPU, 0, &err);
    oskar_telescope_set_position(telescope2,
            longitude_rad, latitude_rad, altitude_m);
    oskar_telescope_set_enable_numerical_patterns(telescope, 0);
    oskar_telescope_load(telescope2, path, NULL, &err);
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

    double max_, avg_;
    oskar_mem_evaluate_relative_error(
            oskar_telescope_station_true_x_enu_metres(telescope),
            oskar_telescope_station_true_x_enu_metres(telescope2),
            0, &max_, &avg_, 0, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    EXPECT_LT(max_, 1e-5);
    EXPECT_LT(avg_, 1e-5);
    oskar_mem_evaluate_relative_error(
            oskar_telescope_station_true_y_enu_metres(telescope),
            oskar_telescope_station_true_y_enu_metres(telescope2),
            0, &max_, &avg_, 0, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    EXPECT_LT(max_, 1e-5);
    EXPECT_LT(avg_, 1e-5);
    oskar_mem_evaluate_relative_error(
            oskar_telescope_station_true_z_enu_metres(telescope),
            oskar_telescope_station_true_z_enu_metres(telescope2),
            0, &max_, &avg_, 0, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    EXPECT_LT(max_, 1e-5);
    EXPECT_LT(avg_, 1e-5);
    for (int i = 0; i < num_stations; ++i)
    {
        oskar_Station* st1 = oskar_telescope_station(telescope, i);
        oskar_Station* st2 = oskar_telescope_station(telescope2, i);
        oskar_mem_evaluate_relative_error(
                oskar_station_element_measured_x_enu_metres(st1),
                oskar_station_element_measured_x_enu_metres(st2),
                0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);
        oskar_mem_evaluate_relative_error(
                oskar_station_element_measured_y_enu_metres(st1),
                oskar_station_element_measured_y_enu_metres(st2),
                0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);
        oskar_mem_evaluate_relative_error(
                oskar_station_element_measured_z_enu_metres(st1),
                oskar_station_element_measured_z_enu_metres(st2),
                0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);
    }

    // Remove test directory.
    oskar_dir_remove(path);

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
            OSKAR_CPU, num_stations, &err);
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
        oskar_telescope_set_station_coords(telescope, i,
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
            oskar_station_set_element_coords(st, j, xyz, xyz, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
            oskar_station_resize(oskar_station_child(st, j), num_elements, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);

            for (int k = 0; k < num_elements; ++k)
            {
                xyz[0] = 100.0 * i + 10.0 * j + 1.0 * k;
                xyz[1] = 200.0 * i + 10.0 * j + 1.0 * k;
                xyz[2] = 300.0 * i + 10.0 * j + 1.0 * k;
                oskar_station_set_element_coords(oskar_station_child(st, j),
                        k, xyz, xyz, &err);
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
    oskar_Telescope* telescope2 = oskar_telescope_create(OSKAR_SINGLE,
                    OSKAR_CPU, 0, &err);
    oskar_telescope_set_position(telescope2,
            longitude_rad, latitude_rad, altitude_m);
    oskar_telescope_set_enable_numerical_patterns(telescope, 0);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    oskar_telescope_load(telescope2, path, NULL, &err);
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

    double max_, avg_ = 0.0;
    oskar_mem_evaluate_relative_error(
            oskar_telescope_station_true_x_enu_metres(telescope),
            oskar_telescope_station_true_x_enu_metres(telescope2),
            0, &max_, &avg_, 0, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    EXPECT_LT(max_, 1e-5);
    EXPECT_LT(avg_, 1e-5);
    oskar_mem_evaluate_relative_error(
            oskar_telescope_station_true_y_enu_metres(telescope),
            oskar_telescope_station_true_y_enu_metres(telescope2),
            0, &max_, &avg_, 0, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    EXPECT_LT(max_, 1e-5);
    EXPECT_LT(avg_, 1e-5);
    oskar_mem_evaluate_relative_error(
            oskar_telescope_station_true_z_enu_metres(telescope),
            oskar_telescope_station_true_z_enu_metres(telescope2),
            0, &max_, &avg_, 0, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    EXPECT_LT(max_, 1e-5);
    EXPECT_LT(avg_, 1e-5);
    for (int i = 0; i < num_stations; ++i)
    {
        oskar_Station* s1 = oskar_telescope_station(telescope, i);
        oskar_Station* s2 = oskar_telescope_station(telescope2, i);
        oskar_mem_evaluate_relative_error(
                oskar_station_element_measured_x_enu_metres(s1),
                oskar_station_element_measured_x_enu_metres(s2),
                0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);
        oskar_mem_evaluate_relative_error(
                oskar_station_element_measured_y_enu_metres(s1),
                oskar_station_element_measured_y_enu_metres(s2),
                0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);
        oskar_mem_evaluate_relative_error(
                oskar_station_element_measured_z_enu_metres(s1),
                oskar_station_element_measured_z_enu_metres(s2),
                0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);

        for (int j = 0; j < num_tiles; ++j)
        {
            oskar_Station *c1 = oskar_station_child(s1, j);
            oskar_Station *c2 = oskar_station_child(s2, j);
            oskar_mem_evaluate_relative_error(
                    oskar_station_element_measured_x_enu_metres(c1),
                    oskar_station_element_measured_x_enu_metres(c2),
                    0, &max_, &avg_, 0, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
            EXPECT_LT(max_, 1e-5);
            EXPECT_LT(avg_, 1e-5);
            oskar_mem_evaluate_relative_error(
                    oskar_station_element_measured_y_enu_metres(c1),
                    oskar_station_element_measured_y_enu_metres(c2),
                    0, &max_, &avg_, 0, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
            EXPECT_LT(max_, 1e-5);
            EXPECT_LT(avg_, 1e-5);
            oskar_mem_evaluate_relative_error(
                    oskar_station_element_measured_z_enu_metres(c1),
                    oskar_station_element_measured_z_enu_metres(c2),
                    0, &max_, &avg_, 0, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
            EXPECT_LT(max_, 1e-5);
            EXPECT_LT(avg_, 1e-5);
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

    oskar_mem_evaluate_relative_error(
            oskar_telescope_station_true_x_enu_metres(telescope),
            oskar_telescope_station_true_x_enu_metres(telescope3),
            0, &max_, &avg_, 0, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    EXPECT_LT(max_, 1e-5);
    EXPECT_LT(avg_, 1e-5);
    oskar_mem_evaluate_relative_error(
            oskar_telescope_station_true_y_enu_metres(telescope),
            oskar_telescope_station_true_y_enu_metres(telescope3),
            0, &max_, &avg_, 0, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    EXPECT_LT(max_, 1e-5);
    EXPECT_LT(avg_, 1e-5);
    oskar_mem_evaluate_relative_error(
            oskar_telescope_station_true_z_enu_metres(telescope),
            oskar_telescope_station_true_z_enu_metres(telescope3),
            0, &max_, &avg_, 0, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    EXPECT_LT(max_, 1e-5);
    EXPECT_LT(avg_, 1e-5);
    for (int i = 0; i < num_stations; ++i)
    {
        oskar_Station* s1 = oskar_telescope_station(telescope, i);
        oskar_Station* s3 = oskar_telescope_station(telescope3, i);
        oskar_mem_evaluate_relative_error(
                oskar_station_element_measured_x_enu_metres(s1),
                oskar_station_element_measured_x_enu_metres(s3),
                0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);
        oskar_mem_evaluate_relative_error(
                oskar_station_element_measured_y_enu_metres(s1),
                oskar_station_element_measured_y_enu_metres(s3),
                0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);
        oskar_mem_evaluate_relative_error(
                oskar_station_element_measured_z_enu_metres(s1),
                oskar_station_element_measured_z_enu_metres(s3),
                0, &max_, &avg_, 0, &err);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);
        EXPECT_LT(max_, 1e-5);
        EXPECT_LT(avg_, 1e-5);

        for (int j = 0; j < num_tiles; ++j)
        {
            oskar_Station *c1 = oskar_station_child(s1, j);
            oskar_Station *c3 = oskar_station_child(s3, j);
            oskar_mem_evaluate_relative_error(
                    oskar_station_element_measured_x_enu_metres(c1),
                    oskar_station_element_measured_x_enu_metres(c3),
                    0, &max_, &avg_, 0, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
            EXPECT_LT(max_, 1e-5);
            EXPECT_LT(avg_, 1e-5);
            oskar_mem_evaluate_relative_error(
                    oskar_station_element_measured_y_enu_metres(c1),
                    oskar_station_element_measured_y_enu_metres(c3),
                    0, &max_, &avg_, 0, &err);
            ASSERT_EQ(0, err) << oskar_get_error_string(err);
            EXPECT_LT(max_, 1e-5);
            EXPECT_LT(avg_, 1e-5);
            oskar_mem_evaluate_relative_error(
                    oskar_station_element_measured_z_enu_metres(c1),
                    oskar_station_element_measured_z_enu_metres(c3),
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
    oskar_dir_remove(path);
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
    int location = OSKAR_CPU;

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
    oskar_telescope_set_position(telescope, 0.1, 0.5, 0.0);
    oskar_telescope_set_enable_numerical_patterns(telescope, 0);
    oskar_telescope_set_enable_noise(telescope, 1, 1);
    QByteArray path = root.toLatin1();

    oskar_telescope_load(telescope, path.constData(), NULL, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);

    ASSERT_EQ(oskar_telescope_num_stations(telescope), num_stations);
    // Check the loaded std.dev. values
    for (int i = 0; i < num_stations; ++i)
    {
        oskar_Station* s = oskar_telescope_station(telescope, i);
        oskar_Mem *freq, *rms;
        freq = oskar_station_noise_freq_hz(s);
        rms = oskar_station_noise_rms_jy(s);
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

    oskar_dir_remove(path.data());
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
        oskar_dir_remove(name_.data());
    }

    root.mkdir(root.absolutePath());

    // Write top-level config file.
    {
        QString config_file = "layout.txt";
        QFile file(dir + QDir::separator() + config_file);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
            return;
        QTextStream out(&file);
        for (int i = 0; i < num_stations; ++i)
            out << "0,0" << endl;
    }

    // Write position file.
    {
        QString config_file = "position.txt";
        QFile file(dir + QDir::separator() + config_file);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
            return;
        QTextStream out(&file);
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
            QString config_file = "layout.txt";
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

