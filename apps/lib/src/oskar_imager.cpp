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
#include "utility/oskar_log_error.h"
#include "utility/oskar_log_message.h"
#include "utility/oskar_log_section.h"
#include "utility/oskar_log_settings.h"
#include "utility/oskar_Log.h"
#include "utility/oskar_Settings.h"
#include "utility/oskar_settings_free.h"

#ifndef OSKAR_NO_FITS
#include "fits/oskar_fits_image_write.h"
#endif

#include <QtCore/QTime>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_imager(const char* settings_file, oskar_Log* log)
{
    int error = 0;
    oskar_Settings settings;
    oskar_Visibilities vis;
    oskar_Image image;

    oskar_log_section(log, "Loading settings file '%s'", settings_file);
    error = oskar_settings_load(&settings, log, settings_file);
    if (error)
    {
        oskar_log_error(log, "Failure in oskar_settings_load() (%s).",
                oskar_get_error_string(error));
        return error;
    }

    // Log the relevant settings.
    log->keep_file = settings.sim.keep_log_file;
    oskar_log_settings_simulator(log, &settings);
    oskar_log_settings_image(log, &settings);

    if (!(settings.image.oskar_image || settings.image.fits_image))
    {
        oskar_log_error(log, "No output image file specified.");
        return OSKAR_ERR_SETTINGS;
    }

    if (!settings.image.input_vis_data)
    {
        oskar_log_error(log, "No input visibility data file specified.");
        return OSKAR_ERR_SETTINGS;
    }

    oskar_visibilities_read(&vis, settings.image.input_vis_data, &error);
    if (error)
    {
        oskar_log_error(log, "Failure in oskar_visibilities_read() (%s).",
                oskar_get_error_string(error));
        return error;
    }

    oskar_log_section(log, "Starting OSKAR imager...");

    QTime timer;
    timer.start();
    error = oskar_make_image(&image, log, &vis, &settings.image);
    if (error)
    {
        oskar_log_error(log, "Failure in oskar_make_image() [code: %i] (%s).",
                error, oskar_get_error_string(error));
        return error;
    }
    oskar_log_section(log, "Imaging completed in %.3f sec.", timer.elapsed()/1.0e3);

    if (settings.image.oskar_image)
    {
        oskar_image_write(&image, log, settings.image.oskar_image, 0, &error);
        if (error)
        {
            oskar_log_error(log, "Failure in oskar_image_write() (%s).",
                    oskar_get_error_string(error));
            return error;
        }
    }
#ifndef OSKAR_NO_FITS
    if (settings.image.fits_image)
    {
        error = oskar_fits_image_write(&image, log, settings.image.fits_image);
        if (error)
        {
            oskar_log_error(log, "Failure in oskar_fits_image_write() (%s).",
                    oskar_get_error_string(error));
            return error;
        }
    }
#endif

    oskar_log_section(log, "Run complete.");
    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
