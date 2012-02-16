/*
 * Copyright (c) 2011, The University of Oxford
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

#include "apps/lib/oskar_set_up_telescope.h"
#include "apps/lib/oskar_load_stations.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_set_value_real.h"
#include "station/oskar_evaluate_station_receiver_noise_stddev.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
#include <algorithm>

using namespace std;

extern "C"
oskar_TelescopeModel* oskar_set_up_telescope(const oskar_Settings* settings)
{
    // Load telescope model into CPU structure.
    oskar_TelescopeModel *telescope;
    const char* telescope_file = settings->telescope.station_positions_file;
    const char* station_dir = settings->telescope.station_layout_directory;
    int type = settings->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    telescope = new oskar_TelescopeModel(type, OSKAR_LOCATION_CPU);
    int err = telescope->load_station_pos(telescope_file,
            settings->telescope.longitude_rad, settings->telescope.latitude_rad,
            settings->telescope.altitude_m);
    if (err)
    {
        fprintf(stderr, "== ERROR: Failed to load telescope geometry (%s).\n",
                oskar_get_error_string(err));
        return NULL;
    }

    // Load stations from directory.
    err = oskar_load_stations(telescope->station,
            &(telescope->identical_stations), telescope->num_stations,
            station_dir);
    if (err)
    {
        fprintf(stderr, "== ERROR: Failed to load station geometry (%s).\n",
                oskar_get_error_string(err));
        return NULL;
    }

    // Find the maximum station size.
    telescope->max_station_size = -INT_MAX;
    for (int i = 0; i < telescope->num_stations; ++i)
    {
        telescope->max_station_size = max(telescope->station[i].num_elements,
                telescope->max_station_size);
    }

    // Set phase centre.
    telescope->ra0_rad = settings->obs.ra0_rad;
    telescope->dec0_rad = settings->obs.dec0_rad;

    // Set other telescope parameters.
    telescope->use_common_sky = true; // FIXME set this via the settings file.
    telescope->bandwidth_hz = settings->obs.channel_bandwidth_hz;
    telescope->wavelength_metres = 0.0; // This is set on a per-channel basis.
    telescope->disable_e_jones = ! (settings->telescope.station.enable_beam);

    // Set other station parameters.
    for (int i = 0; i < telescope->num_stations; ++i)
    {
        telescope->station[i].ra0_rad = telescope->ra0_rad;
        telescope->station[i].dec0_rad = telescope->dec0_rad;
        telescope->station[i].single_element_model = true; // FIXME set this via the settings file.
    }

    // Override station element amplitude gains if required.
    if (settings->telescope.station.element_amp_gain > -1e10)
    {
        for (int i = 0; i < telescope->num_stations; ++i)
        {
            oskar_mem_set_value_real(&telescope->station[i].amp_gain,
                    settings->telescope.station.element_amp_gain);
        }
    }

    // Override station element amplitude errors if required.
    if (settings->telescope.station.element_amp_error > -1e10)
    {
        for (int i = 0; i < telescope->num_stations; ++i)
        {
            oskar_mem_set_value_real(&telescope->station[i].amp_gain_error,
                    settings->telescope.station.element_amp_error);
        }
    }

    // Override station element phase offsets if required.
    if (settings->telescope.station.element_phase_offset_rad > -1e10)
    {
        for (int i = 0; i < telescope->num_stations; ++i)
        {
            oskar_mem_set_value_real(&telescope->station[i].phase_offset,
                    settings->telescope.station.element_phase_offset_rad);
        }
    }

    // Override station element phase errors if required.
    if (settings->telescope.station.element_phase_error_rad > -1e10)
    {
        for (int i = 0; i < telescope->num_stations; ++i)
        {
            oskar_mem_set_value_real(&telescope->station[i].phase_error,
                    settings->telescope.station.element_phase_error_rad);
        }
    }

    // Evaluate station receiver noise (if any specified in the settings)
    if (settings->telescope.station.receiver_temperature_file ||
            settings->telescope.station.receiver_temperature > 0.0)
    {
        int num_channels = settings->obs.num_channels;
        double bandwidth = telescope->bandwidth_hz;
        const oskar_SettingsTime* time = &settings->obs.time;
        double integration_time = time->obs_length_seconds / time->num_vis_dumps;
        vector<double> receiver_temp(num_channels, settings->telescope.station.receiver_temperature);

        // Load receiver temperatures from file.
        if (settings->telescope.station.receiver_temperature_file)
        {
            if (strlen(settings->telescope.station.receiver_temperature_file) > 0)
            {
                //oskar_load_receiver_temperatures(settings->telescope.station.receiver_temperature_file);
                printf("== WARNING: Receiver temperature files are not yet implemented.\n");
            }
        }

        for (int i = 0; i < telescope->num_stations; ++i)
        {
            oskar_StationModel* s = &telescope->station[i];
            oskar_mem_init(&(s->total_receiver_noise), OSKAR_DOUBLE,
                    OSKAR_LOCATION_CPU, num_channels, OSKAR_TRUE);
            oskar_evaluate_station_receiver_noise_stddev(s->total_receiver_noise,
                    &receiver_temp[0], num_channels, bandwidth, integration_time,
                    s->num_elements);
        }
    }

    // Print summary data.
    printf("\n");
    printf("= Telescope model\n");
    printf("  - Num. stations          = %u\n", telescope->num_stations);
    printf("  - Identical stations     = %s\n",
            telescope->identical_stations ? "true" : "false");
    printf("\n");

    // Return the structure.
    return telescope;
}
