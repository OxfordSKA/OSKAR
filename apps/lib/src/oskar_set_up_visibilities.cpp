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

#include "apps/lib/oskar_set_up_visibilities.h"
#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_mem_copy.h"

#include <cstdio>
#include <cstdlib>

extern "C"
oskar_Visibilities* oskar_set_up_visibilities(const oskar_Settings* settings,
        const oskar_TelescopeModel* tel_cpu, int type)
{
    int error;

    // Check the type.
    if (!oskar_mem_is_complex(type))
        return NULL;

    // Create the global visibility structure on the CPU.
    const oskar_SettingsTime* times = &settings->obs.time;
    int n_stations = tel_cpu->num_stations;
    int n_channels = settings->obs.num_channels;
    oskar_Visibilities* vis = new oskar_Visibilities(type, OSKAR_LOCATION_CPU,
            n_channels, times->num_vis_dumps, n_stations * (n_stations - 1) /2);

    // Add meta-data.
    vis->freq_start_hz = settings->obs.start_frequency_hz;
    vis->freq_inc_hz = settings->obs.frequency_inc_hz;
    vis->time_start_mjd_utc = times->obs_start_mjd_utc;
    vis->time_inc_seconds = times->dt_dump_days * 86400.0;
    vis->channel_bandwidth_hz = settings->obs.start_frequency_hz;

    // Add settings file path.
    error = oskar_mem_copy(&vis->settings_path, &settings->settings_path);
    if (error)
    {
        delete vis;
        return NULL;
    }

    // Return the structure.
    return vis;
}
