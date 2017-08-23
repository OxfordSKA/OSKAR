/*
 * Copyright (c) 2013-2016, The University of Oxford
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

#include "apps/oskar_option_parser.h"
#include "apps/oskar_settings_log.h"
#include "apps/oskar_settings_to_telescope.h"
#include "apps/oskar_sim_tec_screen.h"
#include "log/oskar_log.h"
#include "math/oskar_cmath.h"
#include "utility/oskar_version_string.h"
#include "utility/oskar_get_error_string.h"

#include "oskar_settings_load.h"
#include "apps/xml/oskar_sim_tec_screen_xml_all.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <fitsio.h>

using namespace oskar;

static fitsfile* create_fits_file(const char* filename, int precision,
        int width, int height, int num_times, double centre_deg[2],
        double fov_deg[2], double start_time_mjd, double delta_time_sec,
        int* status);
static void write_axis_header(fitsfile* fptr, int axis_id,
        const char* ctype, const char* ctype_comment, double crval,
        double cdelt, double crpix, double crota, int* status);


int main(int argc, char** argv)
{
    int error = 0;

    OptionParser opt("oskar_sim_tec_screen", oskar_version_string());
    opt.add_required("settings file");
    if (!opt.check_options(argc, argv))
        return EXIT_FAILURE;

    oskar_Log* log = oskar_log_create(OSKAR_LOG_MESSAGE, OSKAR_LOG_STATUS);
    oskar_log_message(log, 'M', 0, "Running binary %s", argv[0]);

    const char* settings_file = opt.get_arg(0);
    oskar_Settings_old settings;
    oskar_settings_old_load(&settings, log, settings_file, &error);
    oskar_log_set_keep_file(log, settings.sim.keep_log_file);

    // Get settings.
    const oskar_SettingsIonosphere* MIM = &settings.ionosphere;
    int type = settings.sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    const char* fname = MIM->TECImage.fits_file;
    int im_size = MIM->TECImage.size;
    double fov_deg[2];
    fov_deg[0] = fov_deg[1] = MIM->TECImage.fov_rad * 180. / M_PI;
    int num_times = settings.obs.num_time_steps;
    double t0 = settings.obs.start_mjd_utc;
    double tinc = settings.obs.dt_dump_days * 86400.0;
    if (!fname)
    {
        oskar_log_error(log, "No output file!");
        return EXIT_FAILURE;
    }

    // Run simulation.
    double pp_coord[2];
    oskar_Telescope* tel = oskar_settings_to_telescope(&settings, log, &error);
    oskar_Mem* TEC_screen = oskar_sim_tec_screen(&settings, tel,
            &pp_coord[0], &pp_coord[1], &error);
    pp_coord[0] *= 180. / M_PI;
    pp_coord[1] *= 180. / M_PI;

    // Write TEC screen image.
    if (!error)
    {
        fitsfile* f = create_fits_file(fname, type, im_size, im_size,
                num_times, pp_coord, fov_deg, t0, tinc, &error);
        fits_write_img(f, type == OSKAR_DOUBLE ? TDOUBLE : TFLOAT,
                1, im_size * im_size * num_times,
                oskar_mem_void(TEC_screen), &error);
        fits_close_file(f, &error);
    }
    oskar_mem_free(TEC_screen, &error);
    oskar_telescope_free(tel, &error);

    // Check for errors.
    if (error)
        oskar_log_error(log, "Run failed: %s.", oskar_get_error_string(error));
    oskar_log_free(log);

    return error;
}

static double fov_to_cellsize(double fov_deg, int num_pixels)
{
    double max, inc;
    max = sin(fov_deg * M_PI / 360.0); /* Divide by 2. */
    inc = max / (0.5 * num_pixels);
    return asin(inc) * 180.0 / M_PI;
}

fitsfile* create_fits_file(const char* filename, int precision,
        int width, int height, int num_times, double centre_deg[2],
        double fov_deg[2], double start_time_mjd, double delta_time_sec,
        int* status)
{
    int imagetype;
    long naxes[3];
    double delta;
    fitsfile* f = 0;
    FILE* t = 0;
    if (*status) return 0;

    /* Create a new FITS file and write the image headers. */
    t = fopen(filename, "rb");
    if (t)
    {
        fclose(t);
        remove(filename);
    }
    imagetype = (precision == OSKAR_DOUBLE ? DOUBLE_IMG : FLOAT_IMG);
    naxes[0]  = width;
    naxes[1]  = height;
    naxes[2]  = num_times;
    fits_create_file(&f, filename, status);
    fits_create_img(f, imagetype, 3, naxes, status);
    fits_write_date(f, status);

    /* Write axis headers. */
    delta = fov_to_cellsize(fov_deg[0], width);
    write_axis_header(f, 1, "RA---SIN", "Right Ascension",
            centre_deg[0], -delta, (width + 1) / 2.0, 0.0, status);
    delta = fov_to_cellsize(fov_deg[1], height);
    write_axis_header(f, 2, "DEC--SIN", "Declination",
            centre_deg[1], delta, (height + 1) / 2.0, 0.0, status);
    write_axis_header(f, 3, "UTC", "Time",
            start_time_mjd, delta_time_sec, 1.0, 0.0, status);

    /* Write other headers. */
    fits_write_key_str(f, "TIMESYS", "UTC", NULL, status);
    fits_write_key_str(f, "TIMEUNIT", "s", "Time axis units", status);
    fits_write_key_dbl(f, "MJD-OBS", start_time_mjd, 10, "Start time", status);
    fits_write_key_dbl(f, "OBSRA", centre_deg[0], 10, "RA", status);
    fits_write_key_dbl(f, "OBSDEC", centre_deg[1], 10, "DEC", status);

    return f;
}

void write_axis_header(fitsfile* fptr, int axis_id,
        const char* ctype, const char* ctype_comment, double crval,
        double cdelt, double crpix, double crota, int* status)
{
    char key[FLEN_KEYWORD], value[FLEN_VALUE], comment[FLEN_COMMENT];
    int decimals = 10;
    if (*status) return;

    strncpy(comment, ctype_comment, FLEN_COMMENT-1);
    strncpy(value, ctype, FLEN_VALUE-1);
    fits_make_keyn("CTYPE", axis_id, key, status);
    fits_write_key_str(fptr, key, value, comment, status);
    fits_make_keyn("CRVAL", axis_id, key, status);
    fits_write_key_dbl(fptr, key, crval, decimals, NULL, status);
    fits_make_keyn("CDELT", axis_id, key, status);
    fits_write_key_dbl(fptr, key, cdelt, decimals, NULL, status);
    fits_make_keyn("CRPIX", axis_id, key, status);
    fits_write_key_dbl(fptr, key, crpix, decimals, NULL, status);
    fits_make_keyn("CROTA", axis_id, key, status);
    fits_write_key_dbl(fptr, key, crota, decimals, NULL, status);
}
