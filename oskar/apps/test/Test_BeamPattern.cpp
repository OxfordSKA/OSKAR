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
static const char* app_beam_pattern = "oskar_sim_beam_pattern";

void create_telescope_model(const char* filename, int* status);

TEST(apps_test, test_beam_pattern_modes)
{
    int status = 0;
    printf("OSKAR %s: Testing beam pattern modes...\n", oskar_version_string());

    // Create a telescope model directory.
    const char* tel_model_dir = "apps_test_telescope.tm";
    create_telescope_model(tel_model_dir, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Test single and double precision.
    const char* precs[] = {"single", "double"};
    const int num_precs = sizeof(precs) / sizeof(char*);

    // Test use of CPU and GPU.
    const char* devices[] = {"CPU"
#if defined(OSKAR_HAVE_CUDA) || defined(OSKAR_HAVE_OPENCL)
            , "GPU"
#endif
    };
    const int num_devices = sizeof(devices) / sizeof(char*);

    // Test polarisation modes.
    const char* pol_modes[] = {"Scalar", "Full"};
    const int num_pol_modes = sizeof(pol_modes) / sizeof(char*);

    // Set base parameters.
    const char* sim_par[] = {
            "observation/phase_centre_ra_deg", "20.0",
            "observation/phase_centre_dec_deg", "-30.0",
            "observation/start_frequency_hz", "100e6",
            "observation/num_channels", "1",
            "observation/frequency_inc_hz", "20e6",
            "observation/start_time_utc", "2000-01-01 12:00:00.0",
            "observation/length", "06:00:00.0",
            "observation/num_time_steps", "24",
            "telescope/input_directory", tel_model_dir,
            "telescope/aperture_array/element_pattern/taper/type", "Cosine",
            "telescope/aperture_array/element_pattern/taper/cosine_power", "2",
            "beam_pattern/beam_image/size", "96",
            "beam_pattern/beam_image/fov_deg", "180.0",
            NULL, NULL
    };
    const char* sim_par_grp[][30] = {
            {
                    "beam_pattern/station_outputs/fits_image/auto_power", "true",
                    "beam_pattern/station_outputs/text_file/amp", "true",
                    "beam_pattern/output/average_time_and_channel", "true",
                    "beam_pattern/output/average_single_axis", "Time",
                    "beam_pattern/root_path", "apps_test_bp_station",
                    NULL, NULL
            },
            {
                    "beam_pattern/station_ids", "0,1",
                    "beam_pattern/telescope_outputs/fits_image/cross_power_amp", "true",
                    "beam_pattern/telescope_outputs/fits_image/cross_power_real", "true",
                    "beam_pattern/telescope_outputs/fits_image/cross_power_imag", "true",
                    "beam_pattern/telescope_outputs/text_file/cross_power_amp", "true",
                    "beam_pattern/telescope_outputs/text_file/cross_power_phase", "true",
                    "beam_pattern/output/average_time_and_channel", "true",
                    "beam_pattern/output/average_single_axis", "Time",
                    "beam_pattern/test_source/stokes_i", "true",
                    "beam_pattern/test_source/custom", "true",
                    "beam_pattern/test_source/custom_stokes_i", "2.0",
                    "beam_pattern/root_path", "apps_test_bp_telescope",
                    NULL, NULL
            },
            {
                    "beam_pattern/coordinate_frame", "Horizon",
                    "beam_pattern/station_outputs/fits_image/auto_power", "true",
                    "beam_pattern/root_path", "apps_test_bp_horizon",
                    NULL, NULL
            }
    };
    const int num_par_grp = sizeof(sim_par_grp) / sizeof(char*[30]);

    // Loop over precision.
    for (int i_prec = 0; i_prec < num_precs; ++i_prec)
    {
        // Loop over device types.
        for (int i_dev = 0; i_dev < num_devices; ++i_dev)
        {
            // Loop over polarisation modes.
            for (int i_pol = 0; i_pol < num_pol_modes; ++i_pol)
            {
                // Loop over parameter groups.
                for (int i = 0; i < num_par_grp; ++i)
                {
                    // Create settings and set base values.
                    SettingsTree* sim_settings = oskar_app_settings_tree(
                            app_beam_pattern, 0);
                    ASSERT_TRUE(sim_settings->set_values(0, sim_par));
                    ASSERT_TRUE(sim_settings->set_values(0, sim_par_grp[i]));

                    // Set precision, device and polarisation mode.
                    ASSERT_TRUE(sim_settings->set_value(
                            "simulator/double_precision",
                            !strcmp(precs[i_prec], "single") ? "0" : "1"));
                    ASSERT_TRUE(sim_settings->set_value(
                            "simulator/use_gpus",
                            !strcmp(devices[i_dev], "CPU") ? "0" : "1"));
                    ASSERT_TRUE(sim_settings->set_value(
                            "telescope/pol_mode", pol_modes[i_pol]));

                    // Set output names.
                    string root_name = sim_settings->to_string(
                            "beam_pattern/root_path", &status);
                    root_name += (string("_") + precs[i_prec]);
                    root_name += (string("_") + devices[i_dev]);
                    root_name += (string("_") + pol_modes[i_pol]);
                    ASSERT_TRUE(sim_settings->set_value(
                            "beam_pattern/root_path",
                            string(root_name).c_str()));

                    // Create a beam pattern simulator.
                    oskar_BeamPattern* sim = oskar_settings_to_beam_pattern(
                            sim_settings, 0, &status);
                    ASSERT_EQ(0, status) << oskar_get_error_string(status);

                    // Create a telescope model.
                    oskar_Telescope* tel = oskar_settings_to_telescope(
                            sim_settings, 0, &status);
                    ASSERT_EQ(0, status) << oskar_get_error_string(status);

                    // Run beam pattern simulation.
                    printf("Generating beam pattern: %s\n", root_name.c_str());
                    oskar_beam_pattern_set_telescope_model(sim, tel, &status);
                    ASSERT_EQ(0, status) << oskar_get_error_string(status);
                    oskar_beam_pattern_run(sim, &status);
                    ASSERT_EQ(0, status) << oskar_get_error_string(status);

                    // Delete objects.
                    oskar_beam_pattern_free(sim, &status);
                    oskar_telescope_free(tel, &status);
                    ASSERT_EQ(0, status) << oskar_get_error_string(status);
                    SettingsTree::free(sim_settings);
                }
            }
        }
    }
}
