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

#include <oskar_settings_init.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_settings_init(oskar_Settings* settings)
{
    oskar_SettingsSystemNoise* noise = &settings->interferometer.noise;

    /* Initialise all array pointers to NULL. */
    settings->sim.cuda_device_ids = NULL;
    settings->obs.phase_centre_lon_rad = NULL;
    settings->obs.phase_centre_lat_rad = NULL;
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
    settings->element_fit.input_cst_file = NULL;
    settings->element_fit.input_scalar_file = NULL;
    settings->element_fit.output_directory = NULL;
    settings->element_fit.fits_image = NULL;
    settings->interferometer.ms_filename = NULL;
    settings->interferometer.oskar_vis_filename = NULL;
    settings->beam_pattern.num_active_stations = 0;
    settings->beam_pattern.station_ids = NULL;
    settings->beam_pattern.sky_model = NULL;
    settings->beam_pattern.root_path = NULL;
    settings->beam_pattern.station_fits_ixr = 0;
    settings->beam_pattern.station_fits_phase = 0;
    settings->beam_pattern.station_fits_auto_power_stokes_i = 0;
    settings->beam_pattern.station_fits_amp = 0;
    settings->beam_pattern.telescope_fits_cross_power_stokes_i_amp = 0;
    settings->beam_pattern.size[0] = 0;
    settings->beam_pattern.size[1] = 0;
    settings->image.input_vis_data = NULL;
    settings->image.fits_image = NULL;
    noise->freq.file = NULL;
    noise->rms.file = NULL;
    settings->ionosphere.enable = 0;
    settings->ionosphere.num_TID_screens = 0;
    settings->ionosphere.TID_files = NULL;
    settings->ionosphere.TID = NULL;
    settings->ionosphere.TECImage.fits_file = NULL;
    settings->ionosphere.pierce_points.filename = NULL;

    /* Initialise pathname to settings file. */
    settings->settings_path = NULL;
}

#ifdef __cplusplus
}
#endif
