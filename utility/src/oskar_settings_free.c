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

#include "utility/oskar_settings_free.h"
#include "utility/oskar_mem_free.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_settings_free(oskar_Settings* settings)
{
    int i = 0;

    /* Free all settings arrays. */
    free(settings->sim.cuda_device_ids);
    settings->sim.cuda_device_ids = NULL;
    for (i = 0; i < settings->sky.num_fits_files; ++i)
    {
        free(settings->sky.fits_file[i]);
        settings->sky.fits_file[i] = NULL;
    }
    free(settings->sky.fits_file);
    settings->sky.fits_file = NULL;
    free(settings->sky.gsm_file);
    settings->sky.gsm_file = NULL;
    free(settings->sky.input_sky_file);
    settings->sky.input_sky_file = NULL;
    free(settings->sky.output_sky_file);
    settings->sky.output_sky_file = NULL;
    free(settings->telescope.config_directory);
    settings->telescope.config_directory = NULL;
    free(settings->telescope.output_config_directory);
    settings->telescope.output_config_directory = NULL;
    free(settings->telescope.station.receiver_temperature_file);
    settings->telescope.station.receiver_temperature_file = NULL;
    free(settings->interferometer.ms_filename);
    settings->interferometer.ms_filename = NULL;
    free(settings->interferometer.oskar_vis_filename);
    settings->interferometer.oskar_vis_filename = NULL;
    free(settings->beam_pattern.oskar_image_complex);
    settings->beam_pattern.oskar_image_complex = NULL;
    free(settings->beam_pattern.oskar_image_phase);
    settings->beam_pattern.oskar_image_phase = NULL;
    free(settings->beam_pattern.oskar_image_power);
    settings->beam_pattern.oskar_image_power = NULL;
    free(settings->beam_pattern.fits_image_phase);
    settings->beam_pattern.fits_image_phase = NULL;
    free(settings->beam_pattern.fits_image_power);
    settings->beam_pattern.fits_image_power = NULL;
    free(settings->image.input_vis_data);
    settings->image.input_vis_data = NULL;
    free(settings->image.oskar_image);
    settings->image.oskar_image = NULL;
    free(settings->image.fits_image);
    settings->image.fits_image = NULL;

    /* Free pathname to settings file. */
    oskar_mem_free(&settings->settings_path);

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
