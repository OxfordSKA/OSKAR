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

#define DEG2RAD (M_PI / 180.0)


TEST(Sky, horizon_clip)
{
    int device_loc = 0;
    (void) oskar_device_count(NULL, &device_loc);
    const int locations[] = {OSKAR_CPU, device_loc};
    const int types[] = {OSKAR_SINGLE, OSKAR_DOUBLE};

    // Sky grid parameters.
    const int num_ra = 128, num_dec = 128;
    const int num_sources = num_ra * num_dec;
    const double dec_start_rad = -90. * DEG2RAD, dec_end_rad = 90. * DEG2RAD;
    const double ra_start_rad = 0. * DEG2RAD, ra_end_rad = 330. * DEG2RAD;
    const double ra_range_rad = ra_end_rad - ra_start_rad;
    const double dec_range_rad = dec_end_rad - dec_start_rad;

    // Loop over types and devices.
    for (int i_type = 0; i_type < 2; ++i_type)
    {
        for (int i_dev = 0; i_dev < 2; ++i_dev)
        {
            int status = 0;
            int type = types[i_type];
            int location = locations[i_dev];

            // Generate a grid of sources across the whole sphere.
            oskar_Sky* sky_in = oskar_sky_create(
                    type, OSKAR_CPU, num_sources, &status
            );
            oskar_Mem* col[3];
            for (int i = 0; i < 3; ++i)
            {
                col[i] = oskar_sky_column(
                        sky_in, (oskar_SkyColumn) (i + 1), 0, &status
                );
            }
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            int expected_num_sources = 0;
            for (int i = 0, k = 0; i < num_dec; ++i)
            {
                for (int j = 0; j < num_ra; ++j, ++k)
                {
                    const double ra_rad = (
                            ra_start_rad + j * (ra_range_rad / (num_ra - 1))
                    );
                    const double dec_rad = (
                            dec_start_rad + i * (dec_range_rad / (num_dec - 1))
                    );
                    if (dec_rad >= 0.) expected_num_sources++;
                    oskar_mem_set_element_real(col[0], k, ra_rad, &status);
                    oskar_mem_set_element_real(col[1], k, dec_rad, &status);
                    oskar_mem_set_element_real(col[2], k, (double) k, &status);
                    ASSERT_EQ(0, status) << oskar_get_error_string(status);
                }
            }

            // Evaluate relative direction cosines.
            oskar_sky_evaluate_relative_directions(
                    sky_in, 0.0, M_PI / 2.0, &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);

            // Create a telescope model near the north pole.
            int num_stations = 512;
            oskar_Telescope* telescope = oskar_telescope_create(
                    type, OSKAR_CPU, num_stations, &status
            );
            oskar_telescope_resize_station_array(
                    telescope, num_stations, &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            for (int i = 0; i < num_stations; ++i)
            {
                oskar_station_set_position(
                        oskar_telescope_station(telescope, i),
                        0.0, (90.0 - i * 0.0001) * DEG2RAD, 0.0, 0.0, 0.0, 0.0
                );
            }

            // Create a station work buffer and output sky model.
            oskar_StationWork* work = oskar_station_work_create(
                    type, location, &status
            );
            oskar_Sky* sky_out = oskar_sky_create(type, location, 0, &status);

            // Copy input sky model to device location.
            oskar_Sky* sky_in_dev = oskar_sky_create_copy(
                    sky_in, location, &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);

            // Horizon clip should succeed.
            oskar_Timer* tmr = oskar_timer_create(location);
            oskar_timer_start(tmr);
            oskar_sky_horizon_clip(
                    sky_out, sky_in_dev, telescope, 0.0, work, &status
            );
            printf("Horizon clip took %.3f s\n", oskar_timer_elapsed(tmr));
            oskar_timer_free(tmr);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            const int num_out = oskar_sky_int(sky_out, OSKAR_SKY_NUM_SOURCES);
            EXPECT_EQ(expected_num_sources, num_out);
            printf("Done.\n");

            // Check sky data.
            oskar_Sky* sky_temp = oskar_sky_create_copy(
                    sky_out, OSKAR_CPU, &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            const int num_tmp = oskar_sky_int(sky_temp, OSKAR_SKY_NUM_SOURCES);
            for (int i = 0; i < num_tmp; ++i)
            {
                EXPECT_GE(
                        oskar_sky_data(sky_temp, OSKAR_SKY_DEC_RAD, 0, i), 0.0
                ) << "source " << i;
            }

            // Clean up.
            oskar_sky_free(sky_temp, &status);
            oskar_sky_free(sky_in_dev, &status);
            oskar_sky_free(sky_out, &status);
            oskar_station_work_free(work, &status);
            oskar_sky_free(sky_in, &status);
            oskar_telescope_free(telescope, &status);
        }
    }
}
