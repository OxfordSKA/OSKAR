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
#include "station/oskar_evaluate_station_receiver_noise_stddev.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <QtCore/QByteArray>
using namespace std;

extern "C"
oskar_TelescopeModel* oskar_set_up_telescope(const oskar_Settings& settings)
{
    // Load telescope model into CPU structure.
    oskar_TelescopeModel *telescope;
    QByteArray telescope_file = settings.telescope_file().toAscii();
    QByteArray station_dir = settings.station_dir().toAscii();
    int type = settings.double_precision() ? OSKAR_DOUBLE : OSKAR_SINGLE;
    telescope = new oskar_TelescopeModel(type, OSKAR_LOCATION_CPU);
    int err = telescope->load_station_pos(telescope_file.constData(),
            settings.longitude_rad(), settings.latitude_rad(),
            settings.altitude_m());
    if (err)
    {
        fprintf(stderr, "== ERROR: Failed to load telescope geometry (%s).\n",
                oskar_get_error_string(err));
        return NULL;
    }

    // Load stations from directory.
    err = oskar_load_stations(telescope->station,
            &(telescope->identical_stations), telescope->num_stations,
            station_dir.constData());
    if (err)
    {
        fprintf(stderr, "== ERROR: Failed to load station geometry (%s).\n",
                oskar_get_error_string(err));
        return NULL;
    }

    // Set phase centre.
    telescope->ra0_rad = settings.obs().ra0_rad();
    telescope->dec0_rad = settings.obs().dec0_rad();

    // Set other telescope parameters.
    telescope->use_common_sky = true; // FIXME set this via the settings file.
    telescope->bandwidth_hz = settings.obs().channel_bandwidth();
    telescope->wavelength_metres = 0.0; // This is set on a per-channel basis.
    telescope->disable_e_jones = settings.disable_station_beam();

    // Set other station parameters.
    for (int i = 0; i < telescope->num_stations; ++i)
    {
        telescope->station[i].ra0_rad = telescope->ra0_rad;
        telescope->station[i].dec0_rad = telescope->dec0_rad;
        telescope->station[i].single_element_model = true; // FIXME set this via the settings file.
    }

    // Evaluate station receiver noise (if any specified in the settings)
    if (!settings.receiver_temperature_file().isEmpty() ||
            settings.receiver_temperature() > 0.0)
    {
        int num_channels = settings.obs().num_channels();
        double bandwidth = telescope->bandwidth_hz;
        const oskar_SettingsTime* time =  settings.obs().settings_time();
        double integration_time = time->obs_length_seconds / time->num_vis_dumps;
        vector<double> receiver_temp(num_channels, settings.receiver_temperature());

        // Load receiver temperatures from file.
        if (!settings.receiver_temperature_file().isEmpty())
        {
            //oskar_load_receiver_temperatures(settings.receiver_temperature_file().toAscii().constData());
            printf("== WARNING: Use of receiver temperature files not yet implemented.\n");
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
    printf("= Telescope (%s)\n", telescope_file.constData());
    printf("  - Num. stations          = %u\n", telescope->num_stations);
    printf("  - Identical stations     = %s\n",
            telescope->identical_stations ? "true" : "false");
    printf("\n");

    // Return the structure.
    return telescope;
}
