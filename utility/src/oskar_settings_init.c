/*
 * Copyright (c) 2012-2014, The University of Oxford
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
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_settings_init(oskar_Settings* settings)
{
    int error = 0;
    oskar_SettingsSystemNoise* noise = &settings->interferometer.noise;

    /* Initialise all array pointers to NULL. */
    settings->sim.cuda_device_ids = NULL;
    settings->obs.ra0_rad = NULL;
    settings->obs.dec0_rad = NULL;
    settings->obs.pointing_file = NULL;
    settings->sky.oskar_sky_model.num_files = 0;
    settings->sky.oskar_sky_model.file = NULL;
    settings->sky.gsm.file = NULL;
    settings->sky.output_binary_file = NULL;
    settings->sky.output_text_file = NULL;
    settings->sky.fits_image.num_files = 0;
    settings->sky.fits_image.file = NULL;
    settings->sky.healpix_fits.num_files = 0;
    settings->sky.healpix_fits.file = NULL;
    settings->telescope.input_directory = NULL;
    settings->telescope.output_directory = NULL;
    settings->interferometer.ms_filename = NULL;
    settings->interferometer.oskar_vis_filename = NULL;
    settings->beam_pattern.sky_model = NULL;
    settings->beam_pattern.output_beam_text_file = NULL;
    settings->beam_pattern.oskar_image_voltage = NULL;
    settings->beam_pattern.oskar_image_phase = NULL;
    settings->beam_pattern.oskar_image_complex = NULL;
    settings->beam_pattern.oskar_image_total_intensity = NULL;
    settings->beam_pattern.fits_image_voltage = NULL;
    settings->beam_pattern.fits_image_phase = NULL;
    settings->beam_pattern.fits_image_total_intensity = NULL;
    settings->beam_pattern.size[0] = 0;
    settings->beam_pattern.size[1] = 0;
    settings->image.input_vis_data = NULL;
    settings->image.oskar_image = NULL;
    settings->image.fits_image = NULL;
    noise->freq.file = NULL;
    noise->value.rms.file = NULL;
    noise->value.sensitivity.file = NULL;
    noise->value.t_sys.file = NULL;
    noise->value.area.file = NULL;
    noise->value.efficiency.file = NULL;
    settings->ionosphere.enable = OSKAR_FALSE;
    settings->ionosphere.num_TID_screens = 0;
    settings->ionosphere.TID_files = NULL;
    settings->ionosphere.TID = NULL;
    settings->ionosphere.TECImage.fits_file = NULL;
    settings->ionosphere.TECImage.img_file = NULL;
    settings->ionosphere.pierce_points.filename = NULL;

    /* Initialise pathname to settings file. */
    oskar_mem_init(&settings->settings_path, OSKAR_CHAR,
            OSKAR_LOCATION_CPU, 0, OSKAR_TRUE, &error);

    return error;
}

#ifdef __cplusplus
}
#endif
