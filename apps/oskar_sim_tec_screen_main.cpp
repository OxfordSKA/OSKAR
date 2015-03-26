/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#include <oskar_settings_load.h>
#include <apps/lib/oskar_sim_tec_screen.h>
#include <apps/lib/oskar_OptionParser.h>

#include <oskar_get_error_string.h>
#include <oskar_log.h>
#include <oskar_image.h>
#include <oskar_version_string.h>

#include <fits/oskar_fits_image_write.h>

#include <cstdlib>
#include <cstdio>

int main(int argc, char** argv)
{
    int error = 0;

    oskar_OptionParser opt("oskar_sim_tec_screen", oskar_version_string());
    opt.addRequired("settings file");
    if (!opt.check_options(argc, argv))
        return OSKAR_ERR_INVALID_ARGUMENT;

    oskar_Log* log = oskar_log_create(OSKAR_LOG_MESSAGE, OSKAR_LOG_STATUS);
    oskar_log_list(log, 'M', 0, "Running binary %s", argv[0]);

    const char* settings_file = opt.getArg(0);
    oskar_Settings settings;
    oskar_settings_load(&settings, log, settings_file, &error);
    oskar_log_set_keep_file(log, settings.sim.keep_log_file);

    oskar_log_settings_telescope(log, &settings);
    oskar_log_settings_observation(log, &settings);
    oskar_log_settings_ionosphere(log, &settings);


    // Run simulation.
    oskar_Image* TEC_screen = oskar_sim_tec_screen(&settings, log, &error);

    // Write TEC screen image.
    if (!error)
    {
        const char* fname;
        fname = settings.ionosphere.TECImage.fits_file;
        if (fname && !error)
        {
            oskar_log_list(log, 'M', 0, "Writing FITS image file: '%s'", fname);
            oskar_fits_image_write(TEC_screen, log, fname, &error);
        }
    }
    oskar_image_free(TEC_screen, &error);

    // Check for errors.
    if (error)
        oskar_log_error(log, "Run failed: %s.", oskar_get_error_string(error));
    oskar_log_free(log);

    return error;
}
