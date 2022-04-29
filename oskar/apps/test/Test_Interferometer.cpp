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
static const char* app_imager = "oskar_imager";

void create_sky_model(const char* filename, int* status);
void create_telescope_model(const char* filename, int* status);

TEST(apps, test_interferometer_modes)
{
    int status = 0;
    printf("OSKAR %s: Testing interferometer modes...\n",
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

    // Test using binary file and Measurement Set, if available.
    const char* suffixes[] = {"vis"
#ifndef OSKAR_NO_MS
            , "ms"
#endif
    };
    const int num_suffixes = sizeof(suffixes) / sizeof(char*);

    // Set base parameters.
    string test_name = "apps_test_interferometer_modes";
    const char* sim_par[] = {
            "sky/oskar_sky_model/file", sky_model_file,
            "observation/phase_centre_ra_deg", "20.0",
            "observation/phase_centre_dec_deg", "-30.0",
            "observation/start_frequency_hz", "100e6",
            "observation/num_channels", "3",
            "observation/frequency_inc_hz", "20e6",
            "observation/start_time_utc", "2000-01-01 12:00:00.0",
            "observation/length", "06:00:00.0",
            "observation/num_time_steps", "24",
            "telescope/input_directory", tel_model_dir,
            "telescope/allow_station_beam_duplication", "true",
            "telescope/aperture_array/element_pattern/taper/type", "Cosine",
            "telescope/aperture_array/element_pattern/taper/cosine_power", "2",
            "interferometer/time_average_sec", "1.0",
            "interferometer/correlation_type", "Both",
            "interferometer/noise/enable", "true",
            "interferometer/noise/freq", "Observation settings",
            "interferometer/noise/rms", "Range",
            "interferometer/noise/rms/start", "0.1",
            "interferometer/noise/rms/end", "0.2",
            NULL, NULL
    };
    const char* img_par[] = {
            "image/fov_deg", "2.0",
            "image/size", "128",
            "image/root_path", test_name.c_str(),
            "image/uv_filter_min", "1",
            "image/uv_filter_max", "1e9",
            NULL, NULL
    };

    // Create settings and set base values.
    SettingsTree* sim_settings = oskar_app_settings_tree(app_interferometer, 0);
    ASSERT_TRUE(sim_settings->set_values(0, sim_par));

    // Loop over precision.
    for (int i_prec = 0; i_prec < num_precs; ++i_prec)
    {
        // Loop over device types.
        for (int i_dev = 0; i_dev < num_devices; ++i_dev)
        {
            // Loop over polarisation modes.
            for (int i_pol = 0; i_pol < num_pol_modes; ++i_pol)
            {
                // Set precision, device and polarisation mode.
                ASSERT_TRUE(sim_settings->set_value(
                        "simulator/double_precision",
                        !strcmp(precs[i_prec], "single") ? "0" : "1"));
                ASSERT_TRUE(sim_settings->set_value(
                        "simulator/use_gpus",
                        !strcmp(devices[i_dev], "CPU") ? "0" : "1"));
                ASSERT_TRUE(sim_settings->set_value(
                        "telescope/pol_mode", pol_modes[i_pol]));

                // Set output names for visibilities.
                string root_name = test_name;
                root_name += (string("_") + precs[i_prec]);
                root_name += (string("_") + devices[i_dev]);
                root_name += (string("_") + pol_modes[i_pol]);
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

                // Loop over visibility file types.
                for (int i_suffix = 0; i_suffix < num_suffixes; ++i_suffix)
                {
                    // Create imager settings.
                    SettingsTree* img_settings =
                            oskar_app_settings_tree(app_imager, 0);
                    ASSERT_TRUE(img_settings->set_values(0, img_par));
                    ASSERT_TRUE(img_settings->set_value(
                            "image/double_precision",
                            !strcmp(precs[i_prec], "single") ? "0" : "1"));

                    // Set input visibility file name.
                    string input_vis_data = string(
                            root_name + ".") + suffixes[i_suffix];
                    ASSERT_TRUE(img_settings->set_value(
                            "image/input_vis_data", input_vis_data.c_str()));

                    // Update output image root name.
                    string image_root = img_settings->to_string(
                            "image/root_path", &status);
                    image_root += (string("_") + precs[i_prec]);
                    image_root += (string("_") + devices[i_dev]);
                    image_root += (string("_") + pol_modes[i_pol]);
                    image_root += (string("_") + suffixes[i_suffix]);
                    ASSERT_TRUE(img_settings->set_value(
                            "image/root_path", image_root.c_str()));

                    // Create imager.
                    oskar_Imager* img = oskar_settings_to_imager(
                            img_settings, 0, &status);
                    ASSERT_EQ(0, status) << oskar_get_error_string(status);

                    // Run imager.
                    printf("Generating image: %s\n", image_root.c_str());
                    oskar_imager_run(img, 0, 0, 0, 0, &status);
                    ASSERT_EQ(0, status) << oskar_get_error_string(status);

                    // Delete imager.
                    oskar_imager_free(img, &status);
                    ASSERT_EQ(0, status) << oskar_get_error_string(status);
                    SettingsTree::free(img_settings);
                }
            }
        }
    }

    // Free settings.
    SettingsTree::free(sim_settings);
}
