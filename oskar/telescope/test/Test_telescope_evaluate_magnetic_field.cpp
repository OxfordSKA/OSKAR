/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "telescope/oskar_telescope.h"
#include "utility/oskar_dir.h"
#include "utility/oskar_get_error_string.h"

using std::string;


TEST(telescope_magnetic_field, different_stations)
{
    const char* tel_name = "temp_test_telescope_magnetic_field_different.tm";

    // Create a simple telescope model to load.
    {
        const string tel_dir = string(tel_name) + oskar_dir_separator();
        ASSERT_EQ(1, oskar_dir_mkdir(tel_dir.c_str()));

        // Create the telescope centre position file.
        std::ofstream position(tel_dir + "position.txt");
        position << "116.76, -26.82, 0.0" << std::endl;

        // Create the telescope layout file.
        std::ofstream layout(tel_dir + "layout_wgs84.txt");
        layout << "116.76, -26.82, 0.0" << std::endl;
        layout << "116.26, -26.52, 123.0" << std::endl;
        layout << "116.26, -26.52, 123000.0" << std::endl;
    }

    // Load the telescope model.
    {
        int status = 0;

        // Create an empty telescope model in memory.
        oskar_Telescope* tel = oskar_telescope_create(
                OSKAR_DOUBLE, OSKAR_CPU, 0, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Load the telescope model from the directory.
        oskar_telescope_load(tel, tel_name, 0, &status);
        oskar_telescope_analyse(tel, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, oskar_telescope_num_station_models(tel));

        // Evaluate Earth magnetic field at all stations for a given date.
        oskar_telescope_evaluate_magnetic_field(tel, 2008.25, &status);

        // Check that the magnetic field was evaluated correctly.
        // Comparing here against values output by the IGRF14 reference code
        // at the station positions.
        // Check station 0.
        {
            const oskar_Station* stn = oskar_telescope_station_const(tel, 0);
            const double* field = oskar_station_magnetic_field(stn);
            EXPECT_NEAR(field[0], 140, 0.5);
            EXPECT_NEAR(field[1], 27032, 0.5);
            EXPECT_NEAR(field[2], 48833, 0.5);
            EXPECT_NEAR(field[3], 55816, 0.5);
        }
        // Check station 1.
        {
            const oskar_Station* stn = oskar_telescope_station_const(tel, 1);
            const double* field = oskar_station_magnetic_field(stn);
            EXPECT_NEAR(field[0], 99, 0.5);
            EXPECT_NEAR(field[1], 27186, 0.5);
            EXPECT_NEAR(field[2], 48556, 0.5);
            EXPECT_NEAR(field[3], 55649, 0.5);
        }
        // Check station 2.
        {
            const oskar_Station* stn = oskar_telescope_station_const(tel, 2);
            const double* field = oskar_station_magnetic_field(stn);
            EXPECT_NEAR(field[0], 28, 0.5);
            EXPECT_NEAR(field[1], 25593, 0.5);
            EXPECT_NEAR(field[2], 45584, 0.5);
            EXPECT_NEAR(field[3], 52277, 0.5);
        }

        // Clean up.
        oskar_telescope_free(tel, &status);
    }
    oskar_dir_remove(tel_name);
}


TEST(telescope_magnetic_field, too_far_in_the_future)
{
    const char* tel_name = "temp_test_telescope_magnetic_field_future.tm";

    // Create a simple telescope model to load.
    {
        const string tel_dir = string(tel_name) + oskar_dir_separator();
        ASSERT_EQ(1, oskar_dir_mkdir(tel_dir.c_str()));

        // Create the telescope centre position file.
        std::ofstream position(tel_dir + "position.txt");
        position << "116.76, -26.82, 0.0" << std::endl;

        // Create the telescope layout file.
        std::ofstream layout(tel_dir + "layout_wgs84.txt");
        layout << "116.76, -26.82, 0.0" << std::endl;
    }

    // Load the telescope model.
    {
        int status = 0;

        // Create an empty telescope model in memory.
        oskar_Telescope* tel = oskar_telescope_create(
                OSKAR_DOUBLE, OSKAR_CPU, 0, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Load the telescope model from the directory.
        oskar_telescope_load(tel, tel_name, 0, &status);
        oskar_telescope_analyse(tel, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Try using a date outside the allowed range.
        oskar_telescope_evaluate_magnetic_field(tel, 2100.0, &status);
        ASSERT_EQ(OSKAR_ERR_OUT_OF_RANGE, status);

        // Clean up.
        oskar_telescope_free(tel, &status);
    }
    oskar_dir_remove(tel_name);
}


TEST(telescope_magnetic_field, between_2025_and_2035)
{
    const char* tel_name = "temp_test_telescope_magnetic_field_after_2025.tm";

    // Create a simple telescope model to load.
    {
        const string tel_dir = string(tel_name) + oskar_dir_separator();
        ASSERT_EQ(1, oskar_dir_mkdir(tel_dir.c_str()));

        // Create the telescope centre position file.
        std::ofstream position(tel_dir + "position.txt");
        position << "116.76, -26.82, 0.0" << std::endl;

        // Create the telescope layout file.
        std::ofstream layout(tel_dir + "layout_wgs84.txt");
        layout << "116.76, -26.82, 0.0" << std::endl;
    }

    // Load the telescope model.
    {
        int status = 0;

        // Create an empty telescope model in memory.
        oskar_Telescope* tel = oskar_telescope_create(
                OSKAR_DOUBLE, OSKAR_CPU, 0, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Load the telescope model from the directory.
        oskar_telescope_load(tel, tel_name, 0, &status);
        oskar_telescope_analyse(tel, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Try using a date between 2025 and 2035.
        oskar_telescope_evaluate_magnetic_field(tel, 2028.1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check that the magnetic field was evaluated correctly.
        // Comparing here against values output by the IGRF14 reference code
        // at the station positions.
        // Check station 0.
        {
            const oskar_Station* stn = oskar_telescope_station_const(tel, 0);
            const double* field = oskar_station_magnetic_field(stn);
            EXPECT_NEAR(field[0], 75, 0.5);
            EXPECT_NEAR(field[1], 27812, 0.5);
            EXPECT_NEAR(field[2], 48240, 0.5);
            EXPECT_NEAR(field[3], 55684, 0.5);
        }

        // Clean up.
        oskar_telescope_free(tel, &status);
    }
    oskar_dir_remove(tel_name);
}


TEST(telescope_magnetic_field, before_1995)
{
    const char* tel_name = "temp_test_telescope_magnetic_field_before_1995.tm";

    // Create a simple telescope model to load.
    {
        const string tel_dir = string(tel_name) + oskar_dir_separator();
        ASSERT_EQ(1, oskar_dir_mkdir(tel_dir.c_str()));

        // Create the telescope centre position file.
        std::ofstream position(tel_dir + "position.txt");
        position << "116.76, -26.82, 0.0" << std::endl;

        // Create the telescope layout file.
        std::ofstream layout(tel_dir + "layout_wgs84.txt");
        layout << "116.76, -26.82, 0.0" << std::endl;
    }

    // Load the telescope model.
    {
        int status = 0;

        // Create an empty telescope model in memory.
        oskar_Telescope* tel = oskar_telescope_create(
                OSKAR_DOUBLE, OSKAR_CPU, 0, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Load the telescope model from the directory.
        oskar_telescope_load(tel, tel_name, 0, &status);
        oskar_telescope_analyse(tel, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Try using a date before 1995.
        oskar_telescope_evaluate_magnetic_field(tel, 1991.5, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check that the magnetic field was evaluated correctly.
        // Comparing here against values output by the IGRF14 reference code
        // at the station positions.
        // Check station 0.
        {
            const oskar_Station* stn = oskar_telescope_station_const(tel, 0);
            const double* field = oskar_station_magnetic_field(stn);
            EXPECT_NEAR(field[0], -249, 0.5);
            EXPECT_NEAR(field[1], 26782, 0.5);
            EXPECT_NEAR(field[2], 49358, 0.5);
            EXPECT_NEAR(field[3], 56156, 0.5);
        }

        // Clean up.
        oskar_telescope_free(tel, &status);
    }
    oskar_dir_remove(tel_name);
}
