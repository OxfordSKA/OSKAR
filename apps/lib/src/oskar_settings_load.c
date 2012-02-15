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

#include "apps/lib/oskar_settings_print.h"
#include "apps/lib/oskar_settings_load.h"
#include "apps/lib/oskar_settings_load_observation.h"
#include "apps/lib/oskar_settings_load_simulator.h"
#include "apps/lib/oskar_settings_load_sky.h"
#include "apps/lib/oskar_settings_load_telescope.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_settings_load(oskar_Settings* settings, const char* filename)
{
    int error;

    /* Initialise all array pointers to NULL. */
    settings->obs.ms_filename = 0;
    settings->obs.oskar_vis_filename = 0;
    settings->sim.cuda_device_ids = 0;
    settings->sky.gsm_file = 0;
    settings->sky.input_sky_file = 0;
    settings->sky.output_sky_file = 0;
    settings->telescope.station_positions_file = 0;
    settings->telescope.station_layout_directory = 0;
    settings->telescope.station.receiver_temperature_file = 0;

    error = oskar_settings_load_observation(&settings->obs, filename);
    if (error) return error;
    error = oskar_settings_load_simulator(&settings->sim, filename);
    if (error) return error;
    error = oskar_settings_load_sky(&settings->sky, filename);
    if (error) return error;
    error = oskar_settings_load_telescope(&settings->telescope, filename);
    if (error) return error;

    /* Print settings. */
    oskar_settings_print(settings, filename);

    return 0;
}

#ifdef __cplusplus
}
#endif
