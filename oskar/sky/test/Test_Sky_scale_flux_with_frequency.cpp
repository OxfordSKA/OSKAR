/*
 * Copyright (c) 2011-2026, The OSKAR Developers.
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


TEST(Sky, scale_spectral_index)
{
    int device_loc = 0;
    (void) oskar_device_count(NULL, &device_loc);
    const int num_sources = 10000;
    const int locations[] = {OSKAR_CPU, device_loc};
    const int types[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    const double tolerances[] = {2e-5, 3e-14};
    const double freqs[] = {100e6, 80e6, 60e6, 40e6, 20e6, 5e6};
    const int num_freqs = sizeof(freqs) / sizeof(double);
    const double flux[] = {10.0, 1.0, 0.5, 0.1}; // IQUV.
    const double spx[] = {-0.7, -0.2, 0.1};
    const double freq_ref = 10.0e6;

    for (int i_type = 0; i_type < 2; ++i_type)
    {
        // Create and fill a sky model.
        int status = 0;
        oskar_Sky* sky = oskar_sky_create(
                types[i_type], OSKAR_CPU, num_sources, &status
        );
        for (int i = 0; i < num_sources; ++i)
        {
            const double frac = 2. * M_PI * i / (double) num_sources;
            const double w = sin(frac);
            oskar_sky_set_data(sky, OSKAR_SKY_RA_RAD, 0, i, frac, &status);
            oskar_sky_set_data(sky, OSKAR_SKY_DEC_RAD, 0, i, 0.0, &status);
            oskar_sky_set_data(sky, OSKAR_SKY_I_JY, 0, i, flux[0] * w, &status);
            oskar_sky_set_data(sky, OSKAR_SKY_Q_JY, 0, i, flux[1] * w, &status);
            oskar_sky_set_data(sky, OSKAR_SKY_U_JY, 0, i, flux[2] * w, &status);
            oskar_sky_set_data(sky, OSKAR_SKY_V_JY, 0, i, flux[3] * w, &status);
            oskar_sky_set_data(sky, OSKAR_SKY_REF_HZ, 0, i, freq_ref, &status);
            oskar_sky_set_data(
                    sky, OSKAR_SKY_LIN_SI, 0, i, i % 2 == 0 ? 0. : 1., &status
            );
            if (i < num_sources / 2)
            {
                oskar_sky_set_data(
                        sky, OSKAR_SKY_SPEC_IDX, 0, i, spx[0] * w, &status
                );
                oskar_sky_set_data(
                        sky, OSKAR_SKY_SPEC_IDX, 1, i, spx[1], &status
                );
                oskar_sky_set_data(
                        sky, OSKAR_SKY_SPEC_IDX, 2, i, spx[2], &status
                );
            }
            else
            {
                oskar_sky_set_data(
                        sky, OSKAR_SKY_SPEC_IDX, 0, i, spx[0] * w, &status
                );
            }
        }
        oskar_sky_sort_columns(sky, &status);
        oskar_sky_check_columns(sky, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        // oskar_sky_save_named_columns(
        //         sky, "temp_test_sky_model_spectral_indices.txt",
        //         false, true, true, false, false, false,
        //         &status
        // );

        for (int i_dev = 0; i_dev < 2; ++i_dev)
        {
            // Copy to device.
            oskar_Sky* sky1 = oskar_sky_create_copy(
                    sky, locations[i_dev], &status
            );
            oskar_Timer* timer = oskar_timer_create(locations[i_dev]);

            // Scale at each frequency.
            for (int i_freq = 0; i_freq < num_freqs; ++i_freq)
            {
                const double freq_new = freqs[i_freq];
                oskar_timer_resume(timer);
                oskar_sky_scale_flux_with_frequency(sky1, freq_new, &status);
                oskar_timer_pause(timer);
                ASSERT_EQ(0, status) << oskar_get_error_string(status);

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
                    double factor = 1.;
                    const double w = sin(2. * M_PI * i / (double) num_sources);
                    const double log_r = log10(freq_new / freq_ref);
                    const double base = (freq_new / freq_ref) - 1.0;
                    if (i < num_sources / 2)
                    {
                        if (i % 2 == 0)
                        {
                            // Check a 3-term logarithmic spectral index.
                            factor = pow(
                                    freq_new / freq_ref,
                                    spx[0] * w +
                                    spx[1] * log_r +
                                    spx[2] * log_r * log_r
                            );
                        }
                        else
                        {
                            // Check a 3-term linear spectral index.
                            const double delta = (
                                    spx[0] * w * base +
                                    spx[1] * pow(base, 2.0) +
                                    spx[2] * pow(base, 3.0)
                            );
                            factor = ((flux[0] * w) + delta) / (flux[0] * w);
                        }
                    }
                    else
                    {
                        if (i % 2 == 0)
                        {
                            // Check a single logarithmic spectral index.
                            factor = pow(freq_new / freq_ref, spx[0] * w);
                        }
                        else
                        {
                            // Check a single linear spectral index.
                            const double delta = (spx[0] * w * base);
                            factor = ((flux[0] * w) + delta) / (flux[0] * w);
                        }
                    }
                    for (int c = 0; c < 4; ++c)
                    {
                        oskar_SkyColumn col = (
                                (oskar_SkyColumn) (c + OSKAR_SKY_SCRATCH_I_JY)
                        );
                        ASSERT_NEAR(flux[c] * w * factor,
                                oskar_sky_data(sky2, col, 0, i),
                                tolerances[i_type]
                        ) << "Check failed at i=" << i << " c=" << c;
                    }
                }
                oskar_sky_free(sky2, &status);
            }
            oskar_sky_free(sky1, &status);

            // Report timing.
            printf(
                    "Scale flux with frequency (%s, %s): %.3g sec "
                    "(%d sources, %d frequencies)\n",
                    device_string(locations[i_dev]),
                    oskar_mem_data_type_string(types[i_type]),
                    oskar_timer_elapsed(timer), num_sources, num_freqs
            );
            oskar_timer_free(timer);
        }
        oskar_sky_free(sky, &status);
    }
}


TEST(Sky, scale_flux_with_frequency_no_stokes_i)
{
    int status = 0;
    const int num_sources = 100;
    oskar_Sky* sky = oskar_sky_create(
            OSKAR_DOUBLE, OSKAR_CPU, num_sources, &status
    );
    for (int i = 0; i < num_sources; ++i)
    {
        oskar_sky_set_data(sky, OSKAR_SKY_RA_RAD, 0, i, 1., &status);
        oskar_sky_set_data(sky, OSKAR_SKY_DEC_RAD, 0, i, 2., &status);
        oskar_sky_set_data(sky, OSKAR_SKY_REF_HZ, 0, i, 100e6, &status);
    }
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Scaling should fail, since there is no reference flux.
    oskar_sky_scale_flux_with_frequency(sky, 101e6, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    oskar_sky_free(sky, &status);
}


TEST(Sky, scale_flux_with_frequency_mixed_spectral_types)
{
    int device_loc = 0;
    (void) oskar_device_count(NULL, &device_loc);
    const int locations[] = {OSKAR_CPU, device_loc};
    const int types[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    const int num_freqs = 100;
    const oskar_SkyColumn col = OSKAR_SKY_SCRATCH_I_JY;
    const double freq_start_hz = 95e6;
    const double freq_inc_hz = 100e3;
    const double tolerances[] = {5e-5, 1e-12};

    // Write a test sky model file to load.
    const char* name = "temp_test_mixed_scale_mixed_spectral_types.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "#(Ra, Dec, I, ReferenceFrequency, SpectralIndex, LogarithmicSI, SpectralCurvature, LineWidth) = format\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "# Source 0: Flat spectrum (no reference frequency).\n");
    (void) fprintf(file, "0.00, 0.0, 1.0,,,,,\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "# Source 1: Simple logarithmic spectral index.\n");
    (void) fprintf(file, "0.01, 0.1, 1.1, 101e6, -0.55,,,\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "# Source 2: Two-term logarithmic spectral index polynomial.\n");
    (void) fprintf(file, "0.02, 0.2, 1.2, 102e6, [-0.7, 0.05], true,,\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "# Source 3: Three-term linear spectral index polynomial.\n");
    (void) fprintf(file, "0.03, 0.3, 1.3, 103e6, [0.08, 0.07, 0.02], false,,\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "# Source 4: Spectral curvature model.\n");
    (void) fprintf(file, "0.04, 0.4, 1.4, [104e6], [-0.6],, 0.1,\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "# Source 5: Simple Gaussian spectral line model.\n");
    (void) fprintf(file, "0.05, 0.5, 1.5, 105e6,,,, 100e3\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "# Source 6: Three spectral lines of the same width, each a Gaussian.\n");
    (void) fprintf(file, "0.06, 0.6, [1.6, 1.7, 1.8], [101e6, 102e6, 104e6],,,, 125e3\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "# Source 7: Three spectral lines of different widths, each a Gaussian.\n");
    (void) fprintf(file, "0.07, 0.7, [1.6, 1.7, 1.8], [101e6, 102e6, 104e6],,,, [250e3, 350e3, 500e3]\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "# Source 8: Different flux at four frequencies.\n");
    (void) fprintf(file, "0.08, 0.8, [1.7, 1.8, 1.9, 1.75], [101e6, 102.4e6, 103.8e6, 104.1e6],,,,\n");
    (void) fclose(file);

    for (int i_type = 0; i_type < 2; ++i_type)
    {
        int status = 0;
        const double tol = tolerances[i_type];

        // Load the sky model.
        oskar_Sky* sky = oskar_sky_load_named_columns(
                name, types[i_type], &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        for (int i_dev = 0; i_dev < 2; ++i_dev)
        {
            // Copy sky model to device.
            const int device_loc = locations[i_dev];
            oskar_Sky* sky1 = oskar_sky_create_copy(sky, device_loc, &status);

            // Scale at each frequency.
            for (int i_freq = 0; i_freq < num_freqs; ++i_freq)
            {
                int s = 0;
                double expect = 0.0, freq_ref_hz = 0.0, sigma = 0.0, x = 0.0;
                const double freq_new_hz = freq_start_hz + i_freq * freq_inc_hz;
                oskar_sky_scale_flux_with_frequency(sky1, freq_new_hz, &status);

                // Check Source 0.
                s = 0;
                ASSERT_DOUBLE_EQ(1.0, oskar_sky_data(sky1, col, 0, s));

                // Check Source 1.
                s = 1;
                freq_ref_hz = 101e6;
                expect = 1.1 * pow(freq_new_hz / freq_ref_hz, -0.55);
                ASSERT_NEAR(expect, oskar_sky_data(sky1, col, 0, s), tol);

                // Check Source 2.
                s = 2;
                freq_ref_hz = 102e6;
                const double log_r = log10(freq_new_hz / freq_ref_hz);
                expect = 1.2 * pow(
                        freq_new_hz / freq_ref_hz, -0.7 + 0.05 * log_r
                );
                ASSERT_NEAR(expect, oskar_sky_data(sky1, col, 0, s), tol);

                // Check Source 3.
                s = 3;
                freq_ref_hz = 103e6;
                const double b = (freq_new_hz / freq_ref_hz) - 1.0;
                expect = 1.3 + (0.08 * b) + (0.07 * b * b) + (0.02 * b * b * b);
                ASSERT_NEAR(expect, oskar_sky_data(sky1, col, 0, s), tol);

                // Check Source 4.
                s = 4;
                freq_ref_hz = 104e6;
                expect = 1.4 * pow(freq_new_hz / freq_ref_hz, -0.6) * exp(
                        0.1 * pow(log(freq_new_hz / freq_ref_hz), 2.0)
                );
                ASSERT_NEAR(expect, oskar_sky_data(sky1, col, 0, s), tol);

                // Check Source 5.
                s = 5;
                freq_ref_hz = 105e6;
                sigma = 100e3;
                x = freq_new_hz - freq_ref_hz;
                expect = 1.5 * exp(-x * x / (2 * sigma * sigma));
                ASSERT_NEAR(expect, oskar_sky_data(sky1, col, 0, s), tol);

                // Check Source 6.
                s = 6;
                expect = 0.0;
                freq_ref_hz = 101e6;
                sigma = 125e3;
                x = freq_new_hz - freq_ref_hz;
                expect += 1.6 * exp(-x * x / (2 * sigma * sigma));
                freq_ref_hz = 102e6;
                x = freq_new_hz - freq_ref_hz;
                expect += 1.7 * exp(-x * x / (2 * sigma * sigma));
                freq_ref_hz = 104e6;
                x = freq_new_hz - freq_ref_hz;
                expect += 1.8 * exp(-x * x / (2 * sigma * sigma));
                ASSERT_NEAR(expect, oskar_sky_data(sky1, col, 0, s), tol);

                // Check Source 7.
                s = 7;
                expect = 0.0;
                freq_ref_hz = 101e6;
                sigma = 250e3;
                x = freq_new_hz - freq_ref_hz;
                expect += 1.6 * exp(-x * x / (2 * sigma * sigma));
                freq_ref_hz = 102e6;
                sigma = 350e3;
                x = freq_new_hz - freq_ref_hz;
                expect += 1.7 * exp(-x * x / (2 * sigma * sigma));
                freq_ref_hz = 104e6;
                sigma = 500e3;
                x = freq_new_hz - freq_ref_hz;
                expect += 1.8 * exp(-x * x / (2 * sigma * sigma));
                ASSERT_NEAR(expect, oskar_sky_data(sky1, col, 0, s), tol);

                // Check Source 8.
                s = 8;
                int i_closest = 0;
                x = 1e38;
                for (int j = 0; j < 4; ++j)
                {
                    const double diff = fabs(freq_new_hz - oskar_sky_data(
                            sky1, OSKAR_SKY_REF_HZ, j, s
                    ));
                    if (diff < x)
                    {
                        x = diff;
                        i_closest = j;
                    }
                }
                expect = oskar_sky_data(sky1, OSKAR_SKY_I_JY, i_closest, s);
                ASSERT_NEAR(expect, oskar_sky_data(sky1, col, 0, s), tol);
            }
            oskar_sky_free(sky1, &status);
        }
        oskar_sky_free(sky, &status);
    }
    (void) remove(name);
}


TEST(Sky, scale_flux_with_frequency_rotation_measure)
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

        // Create and fill a sky model.
        oskar_Sky* sky = oskar_sky_create(
                types[i_type], OSKAR_CPU, num_sources, &status
        );
        for (int i = 0; i < num_sources; ++i)
        {
            oskar_sky_set_data(sky, OSKAR_SKY_I_JY, 0, i, 10., &status);
            oskar_sky_set_data(sky, OSKAR_SKY_Q_JY, 0, i, 1., &status);
            oskar_sky_set_data(sky, OSKAR_SKY_U_JY, 0, i, 0., &status);
            oskar_sky_set_data(sky, OSKAR_SKY_V_JY, 0, i, 0.1, &status);
            oskar_sky_set_data(sky, OSKAR_SKY_RM_RAD, 0, i, rm, &status);
            oskar_sky_set_data(sky, OSKAR_SKY_REF_HZ, 0, i, freq_ref, &status);
        }
        for (int i_dev = 0; i_dev < 2; ++i_dev)
        {
            // Copy sky model to device.
            const int device_loc = locations[i_dev];
            oskar_Sky* sky1 = oskar_sky_create_copy(sky, device_loc, &status);

            // Scale with frequency.
            oskar_Timer* timer = oskar_timer_create(device_loc);
            oskar_sky_scale_flux_with_frequency(sky1, freq_new, &status);
            oskar_timer_free(timer);

            // Check that Q is no longer 1, and U no longer 0.
            for (int i = 0; i < num_sources; ++i)
            {
                EXPECT_NE(
                        1., oskar_sky_data(sky1, OSKAR_SKY_SCRATCH_Q_JY, 0, i)
                );
                EXPECT_NE(
                        0., oskar_sky_data(sky1, OSKAR_SKY_SCRATCH_U_JY, 0, i)
                );
            }

            // Scale back to reference frequency.
            oskar_sky_scale_flux_with_frequency(sky1, freq_ref, &status);

            // Check for consistency.
            for (int i = 0; i < num_sources; ++i)
            {
                EXPECT_NEAR(
                        1., oskar_sky_data(sky1, OSKAR_SKY_SCRATCH_Q_JY, 0, i),
                        tolerances[i_type]
                );
                EXPECT_NEAR(
                        0., oskar_sky_data(sky1, OSKAR_SKY_SCRATCH_U_JY, 0, i),
                        tolerances[i_type]
                );
            }

            // Clean up.
            oskar_sky_free(sky1, &status);
        }
        oskar_sky_free(sky, &status);
    }
}
