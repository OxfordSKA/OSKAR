/*
 * Copyright (c) 2012, The University of Oxford
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

#include "apps/lib/oskar_remove_dir.h"
#include "apps/lib/oskar_telescope_model_config_load.h"
#include "apps/lib/oskar_telescope_model_noise_load.h"
#include "apps/lib/oskar_telescope_model_save.h"
#include "apps/lib/test/Test_telescope_model_load_save.h"

#include "interferometry/oskar_TelescopeModel.h"
#include "interferometry/oskar_SettingsTelescope.h"
#include "interferometry/oskar_telescope_model_set_station_coords.h"
#include "station/oskar_StationModel.h"
#include "station/oskar_station_model_init.h"
#include "station/oskar_station_model_resize.h"
#include "station/oskar_station_model_set_element_coords.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_settings_init.h"

#include <QtCore/QtCore>

#include <cstdio>
#include <cstdlib>


void Test_telescope_model_load_save::test_0_level()
{
    int err = 0;
    const char* path = "temp_test_telescope_0_level";
    double longitude_rad = 0.1;
    double latitude_rad = 0.5;
    double altitude_m = 1.0;
    int coord_units = OSKAR_METRES;

    {
        int num_stations = 10;
        //int num_elements = 1;

        // Create a telescope model.
        oskar_TelescopeModel telescope(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_stations);
        telescope.longitude_rad = longitude_rad;
        telescope.latitude_rad  = latitude_rad;
        telescope.altitude_m    = altitude_m;
        telescope.coord_units   = coord_units;

        // Populate the telescope model.
        // TODO?

        err = oskar_telescope_model_save(&telescope, path);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
    }

    // Load it back again.
    {
        oskar_SettingsTelescope settings;
        settings.altitude_m    = altitude_m;
        settings.latitude_rad  = latitude_rad;
        settings.longitude_rad = longitude_rad;
        settings.config_directory = (char*)malloc(1 + strlen(path));
        settings.station.ignore_custom_element_patterns = true;
        strcpy(settings.config_directory, path);

        oskar_TelescopeModel telescope;
        err = oskar_telescope_model_config_load(&telescope, NULL, &settings);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);

        free(settings.config_directory);
    }

    // Remove test directory.
    oskar_remove_dir(path);
}


void Test_telescope_model_load_save::test_1_level()
{
    int err = 0;
    const char* path = "temp_test_telescope_1_level";

    // Create a telescope model.
    int num_stations = 10;
    int num_elements = 20;
    oskar_TelescopeModel telescope(OSKAR_SINGLE, OSKAR_LOCATION_CPU,
            num_stations);
    telescope.longitude_rad = 0.1;
    telescope.latitude_rad = 0.5;
    telescope.altitude_m = 1.0;
    telescope.coord_units = OSKAR_METRES;

    // Populate a telescope model.
    for (int i = 0; i < num_stations; ++i)
    {
        err = oskar_telescope_model_set_station_coords(&telescope, i,
                0.0, 0.0, 0.0, (double) i, (double) (2 * i), (double) (3 * i));
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
        oskar_station_model_resize(&telescope.station[i], num_elements, &err);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);

        for (int j = 0; j < num_elements; ++j)
        {
            err = oskar_station_model_set_element_coords(&telescope.station[i],
                    j, (double) (10 * i + j), (double) (20 * i + j),
                    (double) (30 * i + j), 0.0, 0.0, 0.0);
            CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
        }
    }

    // Save the telescope model.
    err = oskar_telescope_model_save(&telescope, path);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);

    // Load it back again.
    oskar_SettingsTelescope settings;
    settings.altitude_m = telescope.altitude_m;
    settings.latitude_rad = telescope.latitude_rad;
    settings.longitude_rad = telescope.longitude_rad;
    settings.config_directory = (char*)malloc(1 + strlen(path));
    settings.station.ignore_custom_element_patterns = true;
    strcpy(settings.config_directory, path);
    oskar_TelescopeModel telescope2(OSKAR_SINGLE, OSKAR_LOCATION_CPU, 0);
    err = oskar_telescope_model_config_load(&telescope2, NULL, &settings);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
    free(settings.config_directory);

    // Check the contents.
    CPPUNIT_ASSERT_EQUAL(telescope.num_stations, telescope2.num_stations);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(telescope.longitude_rad,
            telescope2.longitude_rad, 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(telescope.latitude_rad,
            telescope2.latitude_rad, 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(telescope.altitude_m,
            telescope2.altitude_m, 1e-10);

    for (int i = 0; i < num_stations; ++i)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(((float*)telescope.station_x_hor)[i],
                ((float*)telescope2.station_x_hor)[i], 1e-5);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(((float*)telescope.station_y_hor)[i],
                ((float*)telescope2.station_y_hor)[i], 1e-5);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(((float*)telescope.station_z_hor)[i],
                ((float*)telescope2.station_z_hor)[i], 1e-5);

        for (int j = 0; j < num_elements; ++j)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((float*)telescope.station[i].x_weights)[j],
                    ((float*)telescope2.station[i].x_weights)[j], 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((float*)telescope.station[i].y_weights)[j],
                    ((float*)telescope2.station[i].y_weights)[j], 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((float*)telescope.station[i].z_weights)[j],
                    ((float*)telescope2.station[i].z_weights)[j], 1e-5);
        }
    }

    // Remove test directory.
    oskar_remove_dir(path);
}


void Test_telescope_model_load_save::test_2_level()
{
    int err = 0;
    const char* path = "temp_test_telescope_2_level";

    // Create a telescope model.
    int num_stations = 3;
    int num_tiles = 4;
    int num_elements = 8;
    oskar_TelescopeModel telescope(OSKAR_SINGLE, OSKAR_LOCATION_CPU,
            num_stations);
    telescope.longitude_rad = 0.1;
    telescope.latitude_rad = 0.5;
    telescope.altitude_m = 1.0;
    telescope.coord_units = OSKAR_METRES;

    // Populate a telescope model.
    for (int i = 0; i < num_stations; ++i)
    {
        err = oskar_telescope_model_set_station_coords(&telescope, i,
                0.0, 0.0, 0.0, (double) i, (double) (2 * i), (double) (3 * i));
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
        oskar_station_model_resize(&telescope.station[i], num_tiles, &err);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
        telescope.station[i].child = (oskar_StationModel*)
                        malloc(sizeof(oskar_StationModel) * num_tiles);

        for (int j = 0; j < num_tiles; ++j)
        {
            err = oskar_station_model_set_element_coords(&telescope.station[i],
                    j, (double) (10 * i + j), (double) (20 * i + j),
                    (double) (30 * i + j), 0.0, 0.0, 0.0);
            CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
            oskar_station_model_init(&telescope.station[i].child[j],
                    OSKAR_SINGLE, OSKAR_LOCATION_CPU, num_elements, &err);
            CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);

            for (int k = 0; k < num_elements; ++k)
            {
                err = oskar_station_model_set_element_coords(&telescope.station[i].child[j],
                        k, (double) (100 * i + 10 * j + k),
                        (double) (200 * i + 20 * j + k),
                        (double) (300 * i + 30 * j + k), 0.0, 0.0, 0.0);
                CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
            }
        }
    }

    // Save the telescope model.
    err = oskar_telescope_model_save(&telescope, path);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);

    // Add an element pattern file.
    QFile file(path + QString("/level0_000/element_pattern_x.txt"));
    file.open(QFile::WriteOnly);
    file.write("\n");
    file.close();

    // Load it back again.
    oskar_SettingsTelescope settings;
    settings.altitude_m = telescope.altitude_m;
    settings.latitude_rad = telescope.latitude_rad;
    settings.longitude_rad = telescope.longitude_rad;
    settings.config_directory = (char*)malloc(1 + strlen(path));
    settings.station.ignore_custom_element_patterns = true;
    strcpy(settings.config_directory, path);
    oskar_TelescopeModel telescope2(OSKAR_SINGLE, OSKAR_LOCATION_CPU, 0);
    err = oskar_telescope_model_config_load(&telescope2, NULL, &settings);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
    free(settings.config_directory);

    // Check the contents.
    CPPUNIT_ASSERT_EQUAL(telescope.num_stations, telescope2.num_stations);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(telescope.longitude_rad,
            telescope2.longitude_rad, 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(telescope.latitude_rad,
            telescope2.latitude_rad, 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(telescope.altitude_m,
            telescope2.altitude_m, 1e-10);

    for (int i = 0; i < num_stations; ++i)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(((float*)telescope.station_x_hor)[i],
                ((float*)telescope2.station_x_hor)[i], 1e-5);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(((float*)telescope.station_y_hor)[i],
                ((float*)telescope2.station_y_hor)[i], 1e-5);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(((float*)telescope.station_z_hor)[i],
                ((float*)telescope2.station_z_hor)[i], 1e-5);

        for (int j = 0; j < num_tiles; ++j)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((float*)telescope.station[i].x_weights)[j],
                    ((float*)telescope2.station[i].x_weights)[j], 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((float*)telescope.station[i].y_weights)[j],
                    ((float*)telescope2.station[i].y_weights)[j], 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((float*)telescope.station[i].z_weights)[j],
                    ((float*)telescope2.station[i].z_weights)[j], 1e-5);

            for (int k = 0; k < num_elements; ++k)
            {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(((float*)telescope.station[i].child[j].x_weights)[k],
                        ((float*)telescope2.station[i].child[j].x_weights)[k], 1e-5);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(((float*)telescope.station[i].child[j].y_weights)[k],
                        ((float*)telescope2.station[i].child[j].y_weights)[k], 1e-5);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(((float*)telescope.station[i].child[j].z_weights)[k],
                        ((float*)telescope2.station[i].child[j].z_weights)[k], 1e-5);
            }
        }
    }

    // Remove test directory.
    oskar_remove_dir(path);
}

//
// TODO: check combinations of telescope model loading and overrides...
//

void Test_telescope_model_load_save::test_load_telescope_noise_rms()
{
    // Test cases that should be considered.
    // -- stddev file at various depths
    // -- number of noise values vs number of stddev
    // -- different modes of getting stddev and freq data.

    QString root = "./temp_test_noise_rms";
    int num_stations = 2;
    int depth = 0;
    int num_values = 5;
    int num_freqs = 5;
    int type = OSKAR_DOUBLE;
    int location = OSKAR_LOCATION_CPU;

    QVector<double> stddev(5);
    for (int i = 0; i < num_values; ++i)
    {
        stddev[i] = i * 0.25 + 0.5;
    }

    QVector<double> freq_values(num_freqs);
    for (int i = 0; i < num_freqs; ++i)
    {
        freq_values[i] = 20.0e6 + i * 10.0e6;
    }

    QHash<QString, QVector<double> > noise_;
    noise_["rms.txt"] = stddev;

    // Generate the telescope
    generate_noisy_telescope(root, num_stations, depth, freq_values, noise_);

    oskar_TelescopeModel telescope(type, location, 0);
    oskar_Settings settings;
    oskar_settings_init(&settings);
    settings.sim.double_precision = (type == OSKAR_DOUBLE) ? OSKAR_TRUE : OSKAR_FALSE;
    QByteArray path = root.toAscii();
    settings.telescope.config_directory = (char*)malloc(root.size() + 1);
    strcpy(settings.telescope.config_directory, path.constData());
    oskar_SettingsSystemNoise* noise = &settings.interferometer.noise;
    noise->enable = OSKAR_TRUE;
    noise->seed = 0;
    noise->value.specification = OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL;
    noise->freq.specification = OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL;
    settings.obs.length_seconds = 1;
    settings.obs.num_time_steps = 1;
    settings.interferometer.channel_bandwidth_hz = 1;


    int err = oskar_telescope_model_noise_load(&telescope, NULL, &settings);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);

    CPPUNIT_ASSERT_EQUAL(telescope.num_stations, num_stations);
    // Check the loaded std.dev. values
    for (int i = 0; i < telescope.num_stations; ++i)
    {
        oskar_SystemNoiseModel* noise = &telescope.station[i].noise;
        int num_values = noise->frequency.num_elements;
        for (int j = 0; j < num_values; ++j)
        {
            if (type == OSKAR_DOUBLE)
            {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        stddev[j],
                        ((double*)noise->rms.data)[j],
                        1.0e-6);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        freq_values[j],
                        ((double*)noise->frequency.data)[j],
                        1.0e-6);
            }
            else
            {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        stddev[j],
                        ((float*)noise->rms.data)[j],
                        1.0e-5);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        freq_values[j],
                        ((float*)noise->frequency.data)[j],
                        1.0e-5);
            }
        }
    }

    oskar_remove_dir(path.data());
}


void Test_telescope_model_load_save::test_load_telescope_noise_sensitivity()
{
    QString root = "./temp_test_noise_sensitivity";
    int num_stations = 2;
    int depth = 1;
    int num_values = 5;
    int num_freqs = 5;
    int type = OSKAR_DOUBLE;
    int location = OSKAR_LOCATION_CPU;

    QVector<double> sensitivity(5);
    for (int i = 0; i < num_values; ++i)
    {
        sensitivity[i] = i * 0.25 + 0.5;
    }

    QVector<double> freq_values(num_freqs);
    for (int i = 0; i < num_freqs; ++i)
    {
        freq_values[i] = 20.0e6 + i * 10.0e6;
    }

    QHash<QString, QVector<double> > noise_;
    noise_["sensitivity.txt"] = sensitivity;

    // Generate the telescope
    generate_noisy_telescope(root, num_stations, depth, freq_values, noise_);

    oskar_TelescopeModel telescope(type, location, 0);
    oskar_Settings settings;
    oskar_settings_init(&settings);
    settings.sim.double_precision = (type == OSKAR_DOUBLE) ? OSKAR_TRUE : OSKAR_FALSE;
    QByteArray path = root.toAscii();
    settings.telescope.config_directory = (char*)malloc(root.size() + 1);
    strcpy(settings.telescope.config_directory, path.constData());
    oskar_SettingsSystemNoise* noise = &settings.interferometer.noise;
    noise->enable = OSKAR_TRUE;
    noise->seed = 0;
    noise->value.specification = OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL;
    noise->freq.specification = OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL;
    settings.obs.length_seconds = 1;
    settings.obs.num_time_steps = 1;
    settings.interferometer.channel_bandwidth_hz = 1;

    double bandwidth = 10.0e6;
    settings.interferometer.channel_bandwidth_hz = bandwidth;
    settings.obs.length_seconds = 60 * 60;
    settings.obs.num_time_steps = 36;
    double integration_time = settings.obs.length_seconds /
            (double)settings.obs.num_time_steps;

    int err = oskar_telescope_model_noise_load(&telescope, NULL, &settings);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);

    CPPUNIT_ASSERT_EQUAL(telescope.num_stations, num_stations);
    // Check the loaded std.dev. values
    for (int i = 0; i < telescope.num_stations; ++i)
    {
        oskar_SystemNoiseModel* noise = &telescope.station[i].noise;
        int num_values = noise->frequency.num_elements;
        for (int j = 0; j < num_values; ++j)
        {
            if (type == OSKAR_DOUBLE)
            {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        sensitivity[j] / sqrt(2.0 * bandwidth * integration_time),
                        ((double*)noise->rms.data)[j],
                        1.0e-6);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        freq_values[j],
                        ((double*)noise->frequency.data)[j],
                        1.0e-6);
            }
            else
            {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        sensitivity[j] / sqrtf(2.0 * bandwidth * integration_time),
                        ((float*)noise->rms.data)[j],
                        1.0e-5);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        freq_values[j],
                        ((float*)noise->frequency.data)[j],
                        1.0e-5);
            }
        }
    }

    oskar_remove_dir(path.data());
}


void Test_telescope_model_load_save::test_load_telescope_noise_t_sys()
{
    QString root = "./temp_test_noise_t_sys";
    int num_stations = 2;
    int depth = 1;
    int num_values = 5;
    int num_freqs = 5;
    int num_areas = 5;
    int type = OSKAR_DOUBLE;
    int location = OSKAR_LOCATION_CPU;

    QVector<double> t_sys(5);
    for (int i = 0; i < num_values; ++i)
    {
        t_sys[i] = i * 0.25 + 0.5;
    }

    QVector<double> freq_values(num_freqs);
    for (int i = 0; i < num_freqs; ++i)
    {
        freq_values[i] = 20.0e6 + i * 10.0e6;
    }

    QVector<double> area_values(num_areas);
    for (int i = 0; i < num_freqs; ++i)
    {
        area_values[i] = i * 5.3 + 1000.0;
    }

    QVector<double> efficiency_values(num_areas);
    for (int i = 0; i < num_freqs; ++i)
    {
        area_values[i] = 0.8;
    }

    QHash<QString, QVector<double> > noise_;
    noise_["t_sys.txt"] = t_sys;
    noise_["area.txt"] = area_values;
    noise_["efficiency.txt"] = efficiency_values;

    // Generate the telescope
    generate_noisy_telescope(root, num_stations, depth, freq_values, noise_);

    oskar_TelescopeModel telescope(type, location, 0);
    oskar_Settings settings;
    oskar_settings_init(&settings);
    settings.sim.double_precision = (type == OSKAR_DOUBLE) ? OSKAR_TRUE : OSKAR_FALSE;
    QByteArray path = root.toAscii();
    settings.telescope.config_directory = (char*)malloc(root.size() + 1);
    strcpy(settings.telescope.config_directory, path.constData());
    oskar_SettingsSystemNoise* noise = &settings.interferometer.noise;
    noise->enable = OSKAR_TRUE;
    noise->seed = 0;
    noise->value.specification = OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL;
    noise->freq.specification = OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL;
    settings.obs.length_seconds = 1;
    settings.obs.num_time_steps = 1;
    settings.interferometer.channel_bandwidth_hz = 1;

    double bandwidth = 10.0e6;
    settings.interferometer.channel_bandwidth_hz = bandwidth;
    settings.obs.length_seconds = 60 * 60;
    settings.obs.num_time_steps = 36;
    double integration_time = settings.obs.length_seconds /
            (double)settings.obs.num_time_steps;

    int err = oskar_telescope_model_noise_load(&telescope, NULL, &settings);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);

    CPPUNIT_ASSERT_EQUAL(telescope.num_stations, num_stations);
    double k_B = 1.3806488e-23;
    double factor = (2.0 * k_B * 1.0e26) / sqrt(2.0 * bandwidth * integration_time);
    // Check the loaded std.dev. values
    for (int i = 0; i < telescope.num_stations; ++i)
    {
        oskar_SystemNoiseModel* noise = &telescope.station[i].noise;
        int num_values = noise->frequency.num_elements;
        for (int j = 0; j < num_values; ++j)
        {
            if (type == OSKAR_DOUBLE)
            {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        (t_sys[j] / (area_values[j] * efficiency_values[j])) * factor,
                        ((double*)noise->rms.data)[j],
                        1.0e-6);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        freq_values[j],
                        ((double*)noise->frequency.data)[j],
                        1.0e-6);
            }
            else
            {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        (t_sys[j] / (area_values[j] * efficiency_values[j])) * factor,
                        ((float*)noise->rms.data)[j],
                        1.0e-5);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(
                        freq_values[j],
                        ((float*)noise->frequency.data)[j],
                        1.0e-5);
            }
        }
    }

    oskar_remove_dir(path.data());
}


void Test_telescope_model_load_save::generate_noisy_telescope(
        const QString& dir, int num_stations, int write_depth,
        const QVector<double>& freqs, const QHash< QString, QVector<double> >& noise)
{
    QDir root(dir);

    if (root.exists())
    {
        QByteArray name_ = dir.toAscii();
        oskar_remove_dir(name_.data());
    }

    root.mkdir(root.absolutePath());

    // Write frequency file.
    if (!freqs.isEmpty())
    {
        QString freq_file = "noise_frequencies.txt";
        QFile file(dir + QDir::separator() + freq_file);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
            return;
        QTextStream out(&file);
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
                    for (int i = 0; i < noise_.value().size(); ++i)
                        out << noise_.value()[i] << endl;
                    ++noise_;
                }
            }
        }
    }
}

