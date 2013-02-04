/*
 * Copyright (c) 2013, The University of Oxford
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
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_settings_free(oskar_Settings* settings)
{
    oskar_SettingsSystemNoise* noise = &settings->interferometer.noise;
    int i = 0, err = 0;

    /* Free all settings arrays. */
    free(settings->sim.cuda_device_ids);
    settings->sim.cuda_device_ids = NULL;
    free(settings->obs.ra0_rad);
    settings->obs.ra0_rad = NULL;
    free(settings->obs.dec0_rad);
    settings->obs.dec0_rad = NULL;

    /* Free pointing file name. */
    free(settings->obs.pointing_file);
    settings->obs.pointing_file = NULL;

    /* Free FITS file names. */
    for (i = 0; i < settings->sky.num_fits_files; ++i)
    {
        free(settings->sky.fits_file[i]);
        settings->sky.fits_file[i] = NULL;
    }
    free(settings->sky.fits_file);
    settings->sky.fits_file = NULL;

    /* Free HEALPix FITS file names. */
    for (i = 0; i < settings->sky.healpix_fits.num_files; ++i)
    {
        free(settings->sky.healpix_fits.file[i]);
        settings->sky.healpix_fits.file[i] = NULL;
    }
    free(settings->sky.healpix_fits.file);
    settings->sky.healpix_fits.file = NULL;

    /* Free GSM file name. */
    free(settings->sky.gsm_file);
    settings->sky.gsm_file = NULL;

    /* Free OSKAR sky model file names. */
    for (i = 0; i < settings->sky.num_sky_files; ++i)
    {
        free(settings->sky.input_sky_file[i]);
        settings->sky.input_sky_file[i] = NULL;
    }
    free(settings->sky.input_sky_file);
    settings->sky.input_sky_file = NULL;

    /* Free output sky model file names. */
    free(settings->sky.output_binary_file);
    settings->sky.output_binary_file = NULL;
    free(settings->sky.output_text_file);
    settings->sky.output_text_file = NULL;

    /* Free telescope directory names. */
    free(settings->telescope.input_directory);
    settings->telescope.input_directory = NULL;
    free(settings->telescope.output_directory);
    settings->telescope.output_directory = NULL;

    /* Free interferometer output file names. */
    free(settings->interferometer.ms_filename);
    settings->interferometer.ms_filename = NULL;
    free(settings->interferometer.oskar_vis_filename);
    settings->interferometer.oskar_vis_filename = NULL;

    /* Free beam pattern file names. */
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

    /* Free imager file names. */
    free(settings->image.input_vis_data);
    settings->image.input_vis_data = NULL;
    free(settings->image.oskar_image);
    settings->image.oskar_image = NULL;
    free(settings->image.fits_image);
    settings->image.fits_image = NULL;

    /* Free noise settings pointers */
    free(noise->freq.file);
    noise->freq.file = NULL;
    free(noise->value.rms.file);
    noise->value.rms.file = NULL;
    free(noise->value.sensitivity.file);
    noise->value.sensitivity.file = NULL;
    free(noise->value.t_sys.file);
    noise->value.t_sys.file = NULL;
    free(noise->value.area.file);
    noise->value.area.file = NULL;
    free(noise->value.efficiency.file);
    noise->value.efficiency.file = NULL;

    /* Free ionosphere settings */
    for (i = 0; i < settings->ionosphere.num_TID_screens; ++i)
    {
        free(settings->ionosphere.TID_files[i]);
        settings->ionosphere.TID_files[i] = NULL;
        free(settings->ionosphere.TID[i].amp);
        settings->ionosphere.TID[i].amp = NULL;
        free(settings->ionosphere.TID[i].speed);
        settings->ionosphere.TID[i].speed = NULL;
        free(settings->ionosphere.TID[i].theta);
        settings->ionosphere.TID[i].theta = NULL;
        free(settings->ionosphere.TID[i].wavelength);
        settings->ionosphere.TID[i].wavelength = NULL;
    }
    free(settings->ionosphere.TID);
    settings->ionosphere.TID = NULL;
    free(settings->ionosphere.TID_files);
    settings->ionosphere.TID_files = NULL;
    free(settings->ionosphere.TECImage.fits_file);
    settings->ionosphere.TECImage.fits_file = NULL;
    free(settings->ionosphere.TECImage.img_file);
    settings->ionosphere.TECImage.img_file = NULL;

    /* Free pathname to settings file. */
    oskar_mem_free(&settings->settings_path, &err);

    return err;
}

#ifdef __cplusplus
}
#endif
