/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <private_telescope.h>
#include <oskar_telescope.h>

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_config_override(oskar_Telescope* telescope,
        const oskar_SettingsTelescope* settings, int* status)
{
    int i;
    const oskar_SettingsArrayElement* array_element =
            &settings->aperture_array.array_pattern.element;

    /* Check all inputs. */
    if (!telescope || !settings || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Override station element systematic/fixed gain errors if required. */
    if (array_element->gain > 0.0 || array_element->gain_error_fixed > 0.0)
    {
        for (i = 0; i < telescope->num_stations; ++i)
        {
            oskar_station_override_element_gains(
                    oskar_telescope_station(telescope, i),
                    array_element->seed_gain_errors, array_element->gain,
                    array_element->gain_error_fixed, status);
        }
    }

    /* Override station element time-variable gain errors if required. */
    if (array_element->gain_error_time > 0.0)
    {
        for (i = 0; i < telescope->num_stations; ++i)
        {
            oskar_station_override_element_time_variable_gains(
                    oskar_telescope_station(telescope, i),
                    array_element->gain_error_time, status);
        }
    }

    /* Override station element systematic/fixed phase errors if required. */
    if (array_element->phase_error_fixed_rad > 0.0)
    {
        for (i = 0; i < telescope->num_stations; ++i)
        {
            oskar_station_override_element_phases(
                    oskar_telescope_station(telescope, i),
                    array_element->seed_phase_errors,
                    array_element->phase_error_fixed_rad, status);
        }
    }

    /* Override station element time-variable phase errors if required. */
    if (array_element->phase_error_time_rad > 0.0)
    {
        for (i = 0; i < telescope->num_stations; ++i)
        {
            oskar_station_override_element_time_variable_phases(
                    oskar_telescope_station(telescope, i),
                    array_element->phase_error_time_rad, status);
        }
    }

    /* Override station element position errors if required. */
    if (array_element->position_error_xy_m > 0.0)
    {
        for (i = 0; i < telescope->num_stations; ++i)
        {
            oskar_station_override_element_xy_position_errors(
                    oskar_telescope_station(telescope, i),
                    array_element->seed_position_xy_errors,
                    array_element->position_error_xy_m, status);
        }
    }

    /* Add variation to x-dipole orientations if required. */
    if (array_element->x_orientation_error_rad > 0.0)
    {
        for (i = 0; i < telescope->num_stations; ++i)
        {
            oskar_station_override_element_orientations(
                    oskar_telescope_station(telescope, i),
                    array_element->seed_x_orientation_error, 1,
                    array_element->x_orientation_error_rad, status);
        }
    }

    /* Add variation to y-dipole orientations if required. */
    if (array_element->y_orientation_error_rad > 0.0)
    {
        for (i = 0; i < telescope->num_stations; ++i)
        {
            oskar_station_override_element_orientations(
                    oskar_telescope_station(telescope, i),
                    array_element->seed_y_orientation_error, 0,
                    array_element->y_orientation_error_rad, status);
        }
    }
}

#ifdef __cplusplus
}
#endif
