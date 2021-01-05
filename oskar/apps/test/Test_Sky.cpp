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

TEST(apps_test, test_sky_model_options)
{
    int status = 0;
    printf("OSKAR %s: Testing sky model options...\n", oskar_version_string());

    // Create a sky model file.
    const char* sky_model_file = "apps_test_sky.txt";
    create_sky_model(sky_model_file, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Test single and double precision.
    const char* precs[] = {"single", "double"};
    const int num_precs = sizeof(precs) / sizeof(char*);

    // Set parameters.
    const char* sim_par[] = {
            "sky/oskar_sky_model/file", sky_model_file,
            "sky/oskar_sky_model/filter/flux_min", "1e-9",
            "sky/oskar_sky_model/filter/flux_max", "1e9",
            "sky/oskar_sky_model/filter/radius_inner_deg", "1e-5",
            "sky/oskar_sky_model/filter/radius_outer_deg", "100.0",
            "sky/generator/random_power_law/num_sources", "10000",
            "sky/generator/random_power_law/flux_min", "1.0",
            "sky/generator/random_power_law/flux_max", "1000.0",
            "sky/generator/random_power_law/power", "-2.0",
            "sky/generator/random_power_law/filter/flux_min", "min",
            "sky/generator/random_power_law/filter/flux_max", "500.0",
            "sky/generator/random_power_law/filter/radius_inner_deg", "10.0",
            "sky/generator/random_power_law/filter/radius_outer_deg", "90.0",
            "sky/generator/random_power_law/extended_sources/FWHM_major", "1.0",
            "sky/generator/random_power_law/extended_sources/FWHM_minor", "0.5",
            "sky/generator/random_broken_power_law/num_sources", "10000",
            "sky/generator/random_broken_power_law/flux_min", "1.0",
            "sky/generator/random_broken_power_law/flux_max", "1000.0",
            "sky/generator/random_broken_power_law/power1", "-1.5",
            "sky/generator/random_broken_power_law/power2", "-2.5",
            "sky/generator/random_broken_power_law/threshold", "100.0",
            "sky/generator/random_broken_power_law/filter/flux_min", "min",
            "sky/generator/random_broken_power_law/filter/flux_max", "500.0",
            "sky/generator/random_broken_power_law/filter/radius_inner_deg", "10.0",
            "sky/generator/random_broken_power_law/filter/radius_outer_deg", "90.0",
            "sky/generator/grid/side_length", "5",
            "sky/generator/grid/fov_deg", "10.0",
            "sky/generator/grid/mean_flux_jy", "0.5",
            "sky/generator/healpix/nside", "4",
            "sky/generator/healpix/filter/radius_inner_deg", "75.0",
            "sky/generator/healpix/filter/radius_outer_deg", "90.0",
            "sky/spectral_index/override", "true",
            "sky/spectral_index/ref_frequency_hz", "100e6",
            "sky/spectral_index/mean", "-0.7",
            "observation/phase_centre_ra_deg", "20.0",
            "observation/phase_centre_dec_deg", "-30.0",
            "observation/start_frequency_hz", "100e6",
            "observation/start_time_utc", "2000-01-01 12:00:00.0",
            "observation/length", "06:00:00.0",
            "observation/num_time_steps", "24",
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

        // Create a sky model.
        oskar_Sky* sky = oskar_settings_to_sky(
                sim_settings, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Print the number of sources.
        printf("Generated sky model contains %d sources.\n",
                oskar_sky_num_sources(sky));

        // Free the sky model.
        oskar_sky_free(sky, &status);
    }

    // Free the settings.
    SettingsTree::free(sim_settings);
}
