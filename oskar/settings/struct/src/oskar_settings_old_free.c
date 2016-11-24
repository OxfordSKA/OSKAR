/*
 * Copyright (c) 2012-2016, The University of Oxford
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

#include "oskar_settings_old_free.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_settings_old_free(oskar_Settings_old* settings)
{
    int i = 0;

    /* Free all settings arrays. */
    free(settings->sim.cuda_device_ids);
    free(settings->obs.phase_centre_lon_rad);
    free(settings->obs.phase_centre_lat_rad);

    /* Free pointing file name. */
    free(settings->obs.pointing_file);

    /* Free FITS file names. */
    for (i = 0; i < settings->sky.fits_image.num_files; ++i)
        free(settings->sky.fits_image.file[i]);
    free(settings->sky.fits_image.file);
    free(settings->sky.fits_image.default_map_units);

    /* Free HEALPix FITS file names. */
    for (i = 0; i < settings->sky.healpix_fits.num_files; ++i)
        free(settings->sky.healpix_fits.file[i]);
    free(settings->sky.healpix_fits.file);
    free(settings->sky.healpix_fits.default_map_units);

    /* Free GSM file name. */
    free(settings->sky.gsm.file);

    /* Free OSKAR sky model file names. */
    for (i = 0; i < settings->sky.oskar_sky_model.num_files; ++i)
        free(settings->sky.oskar_sky_model.file[i]);
    free(settings->sky.oskar_sky_model.file);

    /* Free output sky model file names. */
    free(settings->sky.output_binary_file);
    free(settings->sky.output_text_file);

    /* Free telescope directory names. */
    free(settings->telescope.input_directory);
    free(settings->telescope.output_directory);
    free(settings->telescope.station_type);

    /* Free element fit parameter file names. */
    free(settings->element_fit.input_cst_file);
    free(settings->element_fit.input_scalar_file);
    free(settings->element_fit.output_directory);
    free(settings->element_fit.fits_image);

    /* Free interferometer output file names. */
    free(settings->interferometer.ms_filename);
    free(settings->interferometer.oskar_vis_filename);

    /* Free beam pattern data. */
    free(settings->beam_pattern.station_ids);
    free(settings->beam_pattern.sky_model);
    free(settings->beam_pattern.root_path);

    /* Free noise settings pointers */
    free(settings->interferometer.noise.freq.file);

    /* Free ionosphere settings */
    for (i = 0; i < settings->ionosphere.num_TID_screens; ++i)
    {
        free(settings->ionosphere.TID_files[i]);
        free(settings->ionosphere.TID[i].amp);
        free(settings->ionosphere.TID[i].speed);
        free(settings->ionosphere.TID[i].theta);
        free(settings->ionosphere.TID[i].wavelength);
    }
    free(settings->ionosphere.TID);
    free(settings->ionosphere.TID_files);
    free(settings->ionosphere.TECImage.fits_file);
    free(settings->ionosphere.pierce_points.filename);

    /* Free path name to settings file. */
    free(settings->settings_path);

    /* Clear the structure. */
    memset(settings, 0, sizeof(oskar_Settings_old));
}

#ifdef __cplusplus
}
#endif
