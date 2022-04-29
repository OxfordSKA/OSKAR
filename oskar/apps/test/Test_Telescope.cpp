/*
 * Copyright (c) 2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "apps/oskar_apps.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_version_string.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using oskar::SettingsTree;
using std::string;
static const char* app_interferometer = "oskar_sim_interferometer";

void create_sky_model(const char* filename, int* status);
void create_telescope_model(const char* filename, int* status);

TEST(apps_test, test_telescope_model_options)
{
    int status = 0;
    printf("OSKAR %s: Testing telescope model options...\n",
            oskar_version_string());

    // Create a sky model file.
    const char* sky_model_file = "apps_test_sky.txt";
    create_sky_model(sky_model_file, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Create a telescope model directory.
    const char* tel_model_dir = "apps_test_telescope.tm";
    create_telescope_model(tel_model_dir, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Test single and double precision.
    const char* precs[] = {"single", "double"};
    const int num_precs = sizeof(precs) / sizeof(char*);

    // Set parameters.
    string test_name = "apps_test_telescope_options";
    const char* sim_par[] = {
            "sky/oskar_sky_model/file", sky_model_file,
            "observation/phase_centre_ra_deg", "20.0",
            "observation/phase_centre_dec_deg", "-30.0",
            "observation/start_frequency_hz", "100e6",
            "observation/num_channels", "1",
            "observation/frequency_inc_hz", "20e6",
            "observation/start_time_utc", "2000-01-01 12:00:00.0",
            "observation/length", "06:00:00.0",
            "observation/num_time_steps", "24",
            "telescope/input_directory", tel_model_dir,
            "telescope/aperture_array/array_pattern/element/position_error_xy_m", "0.01",
            "telescope/aperture_array/array_pattern/element/x_gain", "1.1",
            "telescope/aperture_array/array_pattern/element/y_gain", "1.2",
            "telescope/aperture_array/array_pattern/element/x_gain_error_fixed", "0.01",
            "telescope/aperture_array/array_pattern/element/y_gain_error_fixed", "0.02",
            "telescope/aperture_array/array_pattern/element/x_gain_error_time", "0.01",
            "telescope/aperture_array/array_pattern/element/y_gain_error_time", "0.02",
            "telescope/aperture_array/array_pattern/element/x_phase_error_fixed_deg", "5.0",
            "telescope/aperture_array/array_pattern/element/y_phase_error_fixed_deg", "6.0",
            "telescope/aperture_array/array_pattern/element/x_phase_error_time_deg", "1.0",
            "telescope/aperture_array/array_pattern/element/y_phase_error_time_deg", "2.0",
            "telescope/aperture_array/array_pattern/element/x_cable_length_error_m", "0.01",
            "telescope/aperture_array/array_pattern/element/y_cable_length_error_m", "0.02",
            "telescope/aperture_array/array_pattern/element/x_orientation_error_deg", "2.0",
            "telescope/aperture_array/array_pattern/element/y_orientation_error_deg", "3.0",
            NULL, NULL
    };

    // Create settings and set base values.
    SettingsTree* sim_settings = oskar_app_settings_tree(app_interferometer, 0);
    ASSERT_TRUE(sim_settings->set_values(0, sim_par));

    // Loop over precision.
    for (int i_prec = 0; i_prec < num_precs; ++i_prec)
    {
        // Set precision.
        ASSERT_TRUE(sim_settings->set_value(
                "simulator/double_precision",
                !strcmp(precs[i_prec], "single") ? "0" : "1"));

        // Set output names for visibilities.
        string root_name = test_name;
        root_name += (string("_") + precs[i_prec]);
        ASSERT_TRUE(sim_settings->set_value(
                "interferometer/oskar_vis_filename",
                string(root_name + ".vis").c_str()));
#ifndef OSKAR_NO_MS
        ASSERT_TRUE(sim_settings->set_value(
                "interferometer/ms_filename",
                string(root_name + ".ms").c_str()));
#endif

        // Create an interferometer simulator.
        oskar_Interferometer* sim = oskar_settings_to_interferometer(
                sim_settings, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Create a sky model and telescope model.
        oskar_Sky* sky = oskar_settings_to_sky(
                sim_settings, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_Telescope* tel = oskar_settings_to_telescope(
                sim_settings, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Run visibility simulation.
        printf("Generating visibilities: %s\n", root_name.c_str());
        oskar_interferometer_set_telescope_model(sim, tel, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_interferometer_set_sky_model(sim, sky, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_interferometer_run(sim, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Delete objects.
        oskar_interferometer_free(sim, &status);
        oskar_sky_free(sky, &status);
        oskar_telescope_free(tel, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    // Free the settings.
    SettingsTree::free(sim_settings);
}
