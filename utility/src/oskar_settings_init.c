/*
 * Copyright (c) 2012, The University of Oxford
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

#include "utility/oskar_settings_init.h"
#include "utility/oskar_mem_init.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_settings_init(oskar_Settings* settings)
{
    int error = 0;
    oskar_SettingsSystemNoise* noise = &settings->interferometer.noise;

    /* Initialise all array pointers to NULL. */
    settings->sim.cuda_device_ids = NULL;
    settings->sky.num_sky_files = 0;
    settings->sky.input_sky_file = NULL;
    settings->sky.gsm_file = NULL;
    settings->sky.output_sky_file = NULL;
    settings->sky.num_fits_files = 0;
    settings->sky.fits_file = NULL;
    settings->telescope.config_directory = NULL;
    settings->telescope.output_config_directory = NULL;
    settings->telescope.station.receiver_temperature_file = NULL;
    settings->interferometer.ms_filename = NULL;
    settings->interferometer.oskar_vis_filename = NULL;
    settings->beam_pattern.oskar_image_power = NULL;
    settings->beam_pattern.oskar_image_phase = NULL;
    settings->beam_pattern.oskar_image_complex = NULL;
    settings->beam_pattern.fits_image_power = NULL;
    settings->beam_pattern.fits_image_phase = NULL;
    settings->image.input_vis_data = NULL;
    settings->image.oskar_image = NULL;
    settings->image.fits_image = NULL;
    noise->freq.file = NULL;
    noise->value.stddev.file = NULL;
    noise->value.sensitivity.file = NULL;
    noise->value.t_sys.file = NULL;
    noise->value.t_rec.file = NULL;
    noise->value.t_ant.file = NULL;
    noise->value.radiation_efficiency.file = NULL;
    noise->value.area.file = NULL;

    /* Initialise pathname to settings file. */
    error = oskar_mem_init(&settings->settings_path, OSKAR_CHAR,
            OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);
    if (error) return error;

    return error;
}

#ifdef __cplusplus
}
#endif
