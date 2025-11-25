/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "math/oskar_cmath.h"
#include "sky/oskar_sky.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_device.h"
#include "utility/oskar_timer.h"


static const char* device_string(int type)
{
    switch (type)
    {
    case OSKAR_GPU:
        return "CUDA";
    case OSKAR_CL:
        return "OpenCL";
    default:
        break;
    }
    return "CPU";
}


TEST(Sky, scale_by_spectral_index_log_single)
{
    int device_loc = 0;
    (void) oskar_device_count(NULL, &device_loc);
    const int types[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    const double tol[] = {1e-5, 1e-14};
    const int locations[] = {OSKAR_CPU, device_loc};
    const int num_sources = 10000;

    for (int i_type = 0; i_type < 2; ++i_type)
    {
        // Create and fill a sky model.
        int status = 0;
        const double flux[] = {10.0, 1.0, 0.5, 0.1}; // IQUV.
        const double spix = -0.7, freq_ref = 10.0e6, freq_new = 50.0e6;
        oskar_Sky* sky0 = oskar_sky_create(
                types[i_type], OSKAR_CPU, num_sources, &status
        );
        for (int c = 0; c < 4; ++c)
        {
            oskar_SkyColumn col_type = (oskar_SkyColumn) (c + OSKAR_SKY_I_JY);
            oskar_mem_set_value_real(
                    oskar_sky_column(sky0, col_type, 0, &status),
                    flux[c], 0, num_sources, &status
            );
        }
        oskar_mem_set_value_real(
                oskar_sky_column(sky0, OSKAR_SKY_REF_HZ, 0, &status),
                freq_ref, 0, num_sources, &status
        );
        oskar_mem_set_value_real(
                oskar_sky_column(sky0, OSKAR_SKY_SPEC_IDX, 0, &status),
                spix, 0, num_sources, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        for (int i_dev = 0; i_dev < 2; ++i_dev)
        {
            // Copy to device.
            oskar_Sky* sky1 = oskar_sky_create_copy(
                    sky0, locations[i_dev], &status
            );

            // Scale.
            oskar_Timer* timer = oskar_timer_create(locations[i_dev]);
            oskar_timer_resume(timer);
            oskar_sky_scale_flux_with_frequency(sky1, freq_new, &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            printf(
                    "Scale flux with frequency (%s, %s): %.3g sec "
                    "(%d sources)\n",
                    device_string(locations[i_dev]),
                    oskar_mem_data_type_string(types[i_type]),
                    oskar_timer_elapsed(timer), num_sources
            );
            oskar_timer_free(timer);

            // Make copy for checking.
            oskar_Sky* sky2 = oskar_sky_create_copy(
                    sky1, OSKAR_CPU, &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            ASSERT_EQ(num_sources, oskar_sky_int(sky2, OSKAR_SKY_NUM_SOURCES));
            for (int i = 0; i < num_sources; ++i)
            {
                double factor = pow(freq_new / freq_ref, spix);
                for (int c = 0; c < 4; ++c)
                {
                    oskar_SkyColumn col = (
                            (oskar_SkyColumn) (c + OSKAR_SKY_SCRATCH_I_JY)
                    );
                    ASSERT_NEAR(flux[c] * factor,
                            oskar_sky_data(sky2, col, 0, i), tol[i_type]
                    );
                }
            }
            oskar_sky_free(sky2, &status);
            oskar_sky_free(sky1, &status);
        }
        oskar_sky_free(sky0, &status);
    }
}


TEST(Sky, scale_by_spectral_index_log_multi)
{
    int device_loc = 0;
    (void) oskar_device_count(NULL, &device_loc);
    const int num_sources = 10000;
    const int locations[] = {OSKAR_CPU, device_loc};
    const int types[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    const double tol[] = {1e-5, 1e-14};
    const double freqs[] = {50.0e6, 5e6};
    const int num_freqs = sizeof(freqs) / sizeof(double);

    for (int i_type = 0; i_type < 2; ++i_type)
    {
        // Create and fill a sky model.
        int status = 0;
        const double flux[] = {10.0, 1.0, 0.5, 0.1};
        const double spix[] = {-0.7, -0.2, 0.1};
        const double freq_ref = 10.0e6;
        oskar_Sky* sky0 = oskar_sky_create(
                types[i_type], OSKAR_CPU, num_sources, &status
        );
        for (int c = 0; c < 4; ++c)
        {
            oskar_SkyColumn col_type = (oskar_SkyColumn) (c + OSKAR_SKY_I_JY);
            oskar_mem_set_value_real(
                    oskar_sky_column(sky0, col_type, 0, &status),
                    flux[c], 0, num_sources, &status
            );
        }
        oskar_mem_set_value_real(
                oskar_sky_column(sky0, OSKAR_SKY_REF_HZ, 0, &status),
                freq_ref, 0, num_sources, &status
        );
        for (int s = 0; s < 3; ++s)
        {
            oskar_mem_set_value_real(
                    oskar_sky_column(sky0, OSKAR_SKY_SPEC_IDX, s, &status),
                    spix[s], 0, num_sources, &status
            );
        }
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        for (int i_dev = 0; i_dev < 2; ++i_dev)
        {
            // Copy to device.
            oskar_Sky* sky1 = oskar_sky_create_copy(
                    sky0, locations[i_dev], &status
            );

            // Scale at each frequency.
            for (int i_freq = 0; i_freq < num_freqs; ++i_freq)
            {
                // Scale.
                const double freq_new = freqs[i_freq];
                oskar_Timer* timer = oskar_timer_create(locations[i_dev]);
                oskar_timer_resume(timer);
                oskar_sky_scale_flux_with_frequency(sky1, freq_new, &status);
                ASSERT_EQ(0, status) << oskar_get_error_string(status);
                printf(
                        "Scale flux with frequency (%s, %s): %.3g sec "
                        "(%d sources)\n",
                        device_string(locations[i_dev]),
                        oskar_mem_data_type_string(types[i_type]),
                        oskar_timer_elapsed(timer), num_sources
                );
                oskar_timer_free(timer);

                // Make copy for checking.
                oskar_Sky* sky2 = oskar_sky_create_copy(
                        sky1, OSKAR_CPU, &status
                );
                ASSERT_EQ(0, status) << oskar_get_error_string(status);
                ASSERT_EQ(
                        num_sources, oskar_sky_int(sky2, OSKAR_SKY_NUM_SOURCES)
                );
                for (int i = 0; i < num_sources; ++i)
                {
                    const double log_r = log10(freq_new / freq_ref);
                    const double factor = pow(
                            freq_new / freq_ref,
                            spix[0] + spix[1] * log_r + spix[2] * log_r * log_r
                    );
                    for (int c = 0; c < 4; ++c)
                    {
                        oskar_SkyColumn col = (
                                (oskar_SkyColumn) (c + OSKAR_SKY_SCRATCH_I_JY)
                        );
                        ASSERT_NEAR(flux[c] * factor,
                                oskar_sky_data(sky2, col, 0, i), tol[i_type]
                        );
                    }
                }
                oskar_sky_free(sky2, &status);
            }
            oskar_sky_free(sky1, &status);
        }
        oskar_sky_free(sky0, &status);
    }
}


TEST(Sky, scale_by_spectral_index_linear_multi)
{
    int device_loc = 0;
    (void) oskar_device_count(NULL, &device_loc);
    const int num_sources = 10000;
    const int locations[] = {OSKAR_CPU, device_loc};
    const int types[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    const double tol[] = {5e-5, 1e-12};
    const double freqs[] = {50.0e6, 5e6};
    const int num_freqs = sizeof(freqs) / sizeof(double);

    for (int i_type = 0; i_type < 2; ++i_type)
    {
        // Create and fill a sky model.
        int status = 0;
        const double flux[] = {10.0, 1.0, 0.5, 0.1};
        const double spix[] = {-0.7, -0.2, 0.1, -0.05, 0.01, 0.02, -0.01, 0.02};
        const int num_spix_values = sizeof(spix) / sizeof(double);
        const double freq_ref = 10.0e6;
        oskar_Sky* sky0 = oskar_sky_create(
                types[i_type], OSKAR_CPU, num_sources, &status
        );
        for (int c = 0; c < 4; ++c)
        {
            oskar_SkyColumn col_type = (oskar_SkyColumn) (c + OSKAR_SKY_I_JY);
            oskar_mem_set_value_real(
                    oskar_sky_column(sky0, col_type, 0, &status),
                    flux[c], 0, num_sources, &status
            );
        }
        oskar_mem_set_value_real(
                oskar_sky_column(sky0, OSKAR_SKY_REF_HZ, 0, &status),
                freq_ref, 0, num_sources, &status
        );
        oskar_mem_set_value_real(
                oskar_sky_column(sky0, OSKAR_SKY_LIN_SI, 0, &status),
                1.0, 0, num_sources, &status
        );
        for (int s = 0; s < num_spix_values; ++s)
        {
            oskar_mem_set_value_real(
                    oskar_sky_column(sky0, OSKAR_SKY_SPEC_IDX, s, &status),
                    spix[s], 0, num_sources, &status
            );
        }
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        for (int i_dev = 0; i_dev < 2; ++i_dev)
        {
            // Copy to device.
            oskar_Sky* sky1 = oskar_sky_create_copy(
                    sky0, locations[i_dev], &status
            );

            // Scale at each frequency.
            for (int i_freq = 0; i_freq < num_freqs; ++i_freq)
            {
                // Scale.
                const double freq_new = freqs[i_freq];
                oskar_Timer* timer = oskar_timer_create(locations[i_dev]);
                oskar_timer_resume(timer);
                oskar_sky_scale_flux_with_frequency(sky1, freq_new, &status);
                ASSERT_EQ(0, status) << oskar_get_error_string(status);
                printf(
                        "Scale flux with frequency (%s, %s): %.3g sec "
                        "(%d sources)\n",
                        device_string(locations[i_dev]),
                        oskar_mem_data_type_string(types[i_type]),
                        oskar_timer_elapsed(timer), num_sources
                );
                oskar_timer_free(timer);

                // Make copy for checking.
                oskar_Sky* sky2 = oskar_sky_create_copy(
                        sky1, OSKAR_CPU, &status
                );
                ASSERT_EQ(0, status) << oskar_get_error_string(status);
                ASSERT_EQ(
                        num_sources, oskar_sky_int(sky2, OSKAR_SKY_NUM_SOURCES)
                );
                for (int i = 0; i < num_sources; ++i)
                {
                    const double base = (freq_new / freq_ref) - 1.0;
                    const double delta = (
                            spix[0] * base +
                            spix[1] * pow(base, 2.0) +
                            spix[2] * pow(base, 3.0) +
                            spix[3] * pow(base, 4.0) +
                            spix[4] * pow(base, 5.0) +
                            spix[5] * pow(base, 6.0) +
                            spix[6] * pow(base, 7.0) +
                            spix[7] * pow(base, 8.0)
                    );
                    const double factor = (flux[0] + delta) / flux[0];
                    for (int c = 0; c < 4; ++c)
                    {
                        oskar_SkyColumn col = (
                                (oskar_SkyColumn) (c + OSKAR_SKY_SCRATCH_I_JY)
                        );
                        ASSERT_NEAR(flux[c] * factor,
                                oskar_sky_data(sky2, col, 0, i), tol[i_type]
                        );
                    }
                }
                oskar_sky_free(sky2, &status);
            }
            oskar_sky_free(sky1, &status);
        }
        oskar_sky_free(sky0, &status);
    }
}


TEST(Sky, scale_flux_with_frequency_no_stokes_i)
{
    int status = 0;
    const int num_sources = 100;
    oskar_Sky* sky = oskar_sky_create(
            OSKAR_DOUBLE, OSKAR_CPU, num_sources, &status
    );
    oskar_mem_set_value_real(
            oskar_sky_column(sky, OSKAR_SKY_RA_RAD, 0, &status),
            1., 0, num_sources, &status
    );
    oskar_mem_set_value_real(
            oskar_sky_column(sky, OSKAR_SKY_DEC_RAD, 0, &status),
            2., 0, num_sources, &status
    );
    oskar_mem_set_value_real(
            oskar_sky_column(sky, OSKAR_SKY_REF_HZ, 0, &status),
            100e6, 0, num_sources, &status
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Scaling should fail, since there is no reference flux.
    oskar_sky_scale_flux_with_frequency(sky, 101e6, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);

    // Clean up.
    oskar_sky_free(sky, &status);
}


TEST(Sky, scale_flux_with_frequency_too_many_spectral_indices)
{
    int status = 0;
    const int num_sources = 100;
    const int num_spectal_index_values = 9;
    oskar_Sky* sky = oskar_sky_create(
            OSKAR_DOUBLE, OSKAR_CPU, num_sources, &status
    );
    oskar_mem_set_value_real(
            oskar_sky_column(sky, OSKAR_SKY_RA_RAD, 0, &status),
            1., 0, num_sources, &status
    );
    oskar_mem_set_value_real(
            oskar_sky_column(sky, OSKAR_SKY_DEC_RAD, 0, &status),
            2., 0, num_sources, &status
    );
    oskar_mem_set_value_real(
            oskar_sky_column(sky, OSKAR_SKY_I_JY, 0, &status),
            3., 0, num_sources, &status
    );
    oskar_mem_set_value_real(
            oskar_sky_column(sky, OSKAR_SKY_REF_HZ, 0, &status),
            100e6, 0, num_sources, &status
    );
    for (int i = 0; i < num_spectal_index_values; ++i)
    {
        oskar_mem_set_value_real(
                oskar_sky_column(sky, OSKAR_SKY_SPEC_IDX, i, &status),
                (double) i, 0, num_sources, &status
        );
    }
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Scaling should fail.
    oskar_sky_scale_flux_with_frequency(sky, 101e6, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);

    // Clean up.
    oskar_sky_free(sky, &status);
}


TEST(Sky, rotation_measure)
{
    int device_loc = 0;
    (void) oskar_device_count(NULL, &device_loc);
    const int num_sources = 100;
    const int locations[] = {OSKAR_CPU, device_loc};
    const int types[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    const double tolerances[] = {5e-5, 1e-12};
    const double freq_ref = 100.0e6, freq_new = 99e6, rm = 0.5;

    for (int i_type = 0; i_type < 2; ++i_type)
    {
        int status = 0;
        const double tol = tolerances[i_type];

        // Create and fill a sky model.
        oskar_Sky* sky_ref = oskar_sky_create(
                types[i_type], OSKAR_CPU, num_sources, &status
        );
        oskar_mem_set_value_real(
                oskar_sky_column(sky_ref, OSKAR_SKY_I_JY, 0, &status), 10.,
                0, num_sources, &status
        );
        oskar_mem_set_value_real(
                oskar_sky_column(sky_ref, OSKAR_SKY_Q_JY, 0, &status), 1.,
                0, num_sources, &status
        );
        oskar_mem_set_value_real(
                oskar_sky_column(sky_ref, OSKAR_SKY_U_JY, 0, &status), 0.,
                0, num_sources, &status
        );
        oskar_mem_set_value_real(
                oskar_sky_column(sky_ref, OSKAR_SKY_V_JY, 0, &status), 0.1,
                0, num_sources, &status
        );
        oskar_mem_set_value_real(
                oskar_sky_column(sky_ref, OSKAR_SKY_REF_HZ, 0, &status),
                freq_ref, 0, num_sources, &status
        );
        oskar_mem_set_value_real(
                oskar_sky_column(sky_ref, OSKAR_SKY_RM_RAD, 0, &status),
                rm, 0, num_sources, &status
        );

        for (int i_dev = 0; i_dev < 2; ++i_dev)
        {
            // Copy sky model to device.
            const int device_loc = locations[i_dev];
            oskar_Sky* sky_dev = oskar_sky_create_copy(
                    sky_ref, device_loc, &status
            );

            // Scale with frequency.
            oskar_Timer* timer = oskar_timer_create(device_loc);
            oskar_sky_scale_flux_with_frequency(sky_dev, freq_new, &status);
            oskar_timer_free(timer);

            // Check that Q is no longer 1, and U no longer 0.
            for (int i = 0; i < num_sources; ++i)
            {
                EXPECT_NE(
                        1., oskar_sky_data(
                                sky_dev, OSKAR_SKY_SCRATCH_Q_JY, 0, i
                        )
                );
                EXPECT_NE(
                        0., oskar_sky_data(
                                sky_dev, OSKAR_SKY_SCRATCH_U_JY, 0, i
                        )
                );
            }

            // Scale back to reference frequency.
            oskar_sky_scale_flux_with_frequency(sky_dev, freq_ref, &status);

            // Check for consistency.
            for (int i = 0; i < num_sources; ++i)
            {
                EXPECT_NEAR(
                        1., oskar_sky_data(
                                sky_dev, OSKAR_SKY_SCRATCH_Q_JY, 0, i
                        ), tol
                );
                EXPECT_NEAR(
                        0., oskar_sky_data(
                                sky_dev, OSKAR_SKY_SCRATCH_U_JY, 0, i
                        ), tol
                );
            }

            // Clean up.
            oskar_sky_free(sky_dev, &status);
        }
        oskar_sky_free(sky_ref, &status);
    }
}
