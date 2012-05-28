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
#include "apps/lib/oskar_telescope_model_load.h"
#include "apps/lib/oskar_telescope_model_save.h"
#include "apps/lib/test/Test_telescope_model_load_save.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "interferometry/oskar_SettingsTelescope.h"
#include "interferometry/oskar_telescope_model_set_station_coords.h"
#include "station/oskar_StationModel.h"
#include "station/oskar_station_model_init.h"
#include "station/oskar_station_model_set_element_coords.h"
#include "utility/oskar_get_error_string.h"

#include <QtCore/QFile>
#include <cstdio>
#include <cstdlib>

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
        err = telescope.station[i].resize(num_elements);
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
    err = oskar_telescope_model_load(&telescope2, NULL, &settings);
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
        err = telescope.station[i].resize(num_tiles);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
        telescope.station[i].child = (oskar_StationModel*)
                malloc(sizeof(oskar_StationModel) * num_tiles);

        for (int j = 0; j < num_tiles; ++j)
        {
            err = oskar_station_model_set_element_coords(&telescope.station[i],
                    j, (double) (10 * i + j), (double) (20 * i + j),
                    (double) (30 * i + j), 0.0, 0.0, 0.0);
            CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
            err = oskar_station_model_init(&telescope.station[i].child[j],
                    OSKAR_SINGLE, OSKAR_LOCATION_CPU, num_elements);
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
    err = oskar_telescope_model_load(&telescope2, NULL, &settings);
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
