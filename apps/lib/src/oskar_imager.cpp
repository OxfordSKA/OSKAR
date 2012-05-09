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

#include "apps/lib/oskar_imager.h"
#include "apps/lib/oskar_settings_load.h"
#include "imaging/oskar_Image.h"
#include "imaging/oskar_make_image.h"
#include "imaging/oskar_image_write.h"
#include "interferometry/oskar_Visibilities.h"
#include "interferometry/oskar_visibilities_read.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_Settings.h"
#include "utility/oskar_settings_free.h"

#ifndef OSKAR_NO_FITS
#include "fits/oskar_fits_image_write.h"
#endif

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_imager(const char* settings_file)
{
    int error;
    oskar_Settings settings;
    oskar_Visibilities vis;
    oskar_Image image;

    /* Note: this could probably be replaced with oskar_settings_load_image() */
    error = oskar_settings_load(&settings, settings_file);
    if (error)
    {
        fprintf(stderr, "\nERROR: oskar_settings_load() failed!, %s\n",
                oskar_get_error_string(error));
        return error;
    }

    if (!(settings.image.oskar_image || settings.image.fits_image))
    {
        fprintf(stderr, "ERROR: No output image file specified in the settings.\n");
        return OSKAR_ERR_SETTINGS;
    }

    if (!settings.image.input_vis_data)
    {
        fprintf(stderr, "ERROR: No input visibility data specified in the settings.\n");
        return OSKAR_ERR_SETTINGS;
    }

    error = oskar_visibilities_read(&vis, settings.image.input_vis_data);
    if (error)
    {
        fprintf(stderr, "\nERROR: oskar_Visibilities::read() failed!, %s\n",
                oskar_get_error_string(error));
        return error;
    }

    fprintf(stdout, "=== Starting OSKAR imager...\n");

    error = oskar_make_image(&image, &vis, &settings.image);
    if (error)
    {
        fprintf(stderr, "\nERROR: oskar_make_image() failed!, %s\n",
                oskar_get_error_string(error));
        return error;
    }
    fprintf(stdout, "=== Imaging complete.\n\n");

    if (settings.image.oskar_image)
    {
        printf("--> Writing OSKAR image: '%s'\n", settings.image.oskar_image);
        error = oskar_image_write(&image, settings.image.oskar_image, 0);
        if (error)
        {
            fprintf(stderr, "\nERROR: oskar_image_write() failed!, %s\n",
                    oskar_get_error_string(error));
            return error;
        }
    }
#ifndef OSKAR_NO_FITS
    if (settings.image.fits_image)
    {
        printf("--> Writing FITS image: '%s'\n", settings.image.fits_image);
        /* Note: currently there is no error code returned from this function. */
        error = oskar_fits_image_write(&image, settings.image.fits_image);
        if (error)
        {
            fprintf(stderr, "\nERROR: oskar_fits_image_write() failed!, %s\n",
                    oskar_get_error_string(error));
            return error;
        }
    }
#endif

    fprintf(stdout, "\n=== Run complete.\n");
    oskar_settings_free(&settings);
    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
