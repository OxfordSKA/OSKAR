/*
 * Copyright (c) 2011-2016, The University of Oxford
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

#include <oskar_set_up_sky.h>
#include <oskar_convert_brightness_to_jy.h>
#include <oskar_convert_healpix_ring_to_theta_phi.h>
#include <oskar_healpix_npix_to_nside.h>
#include <oskar_random_gaussian.h>
#include <oskar_random_broken_power_law.h>
#include <oskar_generate_random_coordinate.h>
#include <oskar_sky.h>
#include <oskar_log.h>
#include <oskar_get_error_string.h>

#include <oskar_cmath.h>
#include <stdlib.h> /* For srand() */
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static void set_up_osm(oskar_Sky* sky, oskar_Log* log,
        const oskar_SettingsSkyOskar* s, double ra0, double dec0, int* status);
static void set_up_gsm(oskar_Sky* sky, oskar_Log* log,
        const oskar_SettingsSkyGsm* s, double ra0, double dec0, int* status);
static void set_up_fits_image(oskar_Sky* sky, oskar_Log* log,
        const oskar_SettingsSkyFitsImage* s, double ra0, double dec0,
        int* status);
static void set_up_healpix_fits(oskar_Sky* sky, oskar_Log* log,
        const oskar_SettingsSkyHealpixFits* s, double ra0, double dec0,
        int* status);

static void set_up_gen_grid(oskar_Sky* sky, oskar_Log* log,
        const oskar_SettingsSkyGeneratorGrid* s, double ra0, double dec0,
        int* status);
static void set_up_gen_healpix(oskar_Sky* sky, oskar_Log* log,
        const oskar_SettingsSkyGeneratorHealpix* s, double ra0, double dec0,
        int* status);
static void set_up_gen_rpl(oskar_Sky* sky, oskar_Log* log,
        const oskar_SettingsSkyGeneratorRandomPowerLaw* s, double ra0,
        double dec0, int* status);
static void set_up_gen_rbpl(oskar_Sky* sky, oskar_Log* log,
        const oskar_SettingsSkyGeneratorRandomBrokenPowerLaw* s, double ra0,
        double dec0, int* status);

static void set_up_filter(oskar_Sky* sky, const oskar_SettingsSkyFilter* f,
        double ra0_rad, double dec0_rad, int* status);
static void set_up_extended(oskar_Sky* sky,
        const oskar_SettingsSkyExtendedSources* ext, int* status);
static void set_up_pol(oskar_Sky* sky,
        const oskar_SettingsSkyPolarisation* pol, int* status);

oskar_Sky* oskar_set_up_sky(const oskar_Settings_old* settings, oskar_Log* log,
        int* status)
{
    int type, i, num_sources;
    double ra0, dec0;
    oskar_Sky* sky = 0;
    const char* filename;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Sky model data type and settings. */
    oskar_log_section(log, 'M', "Sky model");
    type = settings->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    sky = oskar_sky_create(type, OSKAR_CPU, 0, status);
    ra0  = settings->obs.phase_centre_lon_rad[0];
    dec0 = settings->obs.phase_centre_lat_rad[0];

    /* Load sky model data files. */
    set_up_osm(sky, log, &settings->sky.oskar_sky_model, ra0, dec0, status);
    set_up_gsm(sky, log, &settings->sky.gsm, ra0, dec0, status);
    set_up_fits_image(sky, log, &settings->sky.fits_image, ra0, dec0, status);
    set_up_healpix_fits(sky, log, &settings->sky.healpix_fits,
            ra0, dec0, status);

    /* Generate sky models from generator parameters. */
    set_up_gen_grid(sky, log, &settings->sky.generator.grid,
            ra0, dec0, status);
    set_up_gen_healpix(sky, log, &settings->sky.generator.healpix,
            ra0, dec0, status);
    set_up_gen_rpl(sky, log, &settings->sky.generator.random_power_law,
            ra0, dec0, status);
    set_up_gen_rbpl(sky, log, &settings->sky.generator.random_broken_power_law,
            ra0, dec0, status);

    /* Return if sky model contains no sources. */
    num_sources = oskar_sky_num_sources(sky);
    if (num_sources == 0)
    {
        oskar_log_warning(log, "Sky model contains no sources.");
        return sky;
    }

    /* Perform final overrides. */
    if (settings->sky.spectral_index.override)
    {
        double mean, std_dev, ref_freq, val[2];
        mean = settings->sky.spectral_index.mean;
        std_dev = settings->sky.spectral_index.std_dev;
        ref_freq = settings->sky.spectral_index.ref_frequency_hz;

        /* Override source spectral index values. */
        oskar_log_message(log, 'M', 0, "Overriding source spectral index values...");
        for (i = 0; i < num_sources; ++i)
        {
            oskar_random_gaussian2(settings->sky.spectral_index.seed,
                    i, 0, val);
            val[0] = std_dev * val[0] + mean;
            oskar_sky_set_spectral_index(sky, i, ref_freq, val[0], status);
        }
        oskar_log_message(log, 'M', 1, "done.");
    }

    if (*status) return sky;

    /* Print summary data. */
    oskar_log_message(log, 'M', 0, "Sky model summary");
    oskar_log_value(log, 'M', 1, "Num. sources", "%d", num_sources);

    /* Write text file. */
    filename = settings->sky.output_text_file;
    if (filename && strlen(filename) && !*status)
    {
        oskar_log_message(log, 'M', 1, "Writing sky model text file: %s", filename);
        oskar_sky_save(filename, sky, status);
    }

    /* Write binary file. */
    filename = settings->sky.output_binary_file;
    if (filename && strlen(filename) && !*status)
    {
        oskar_log_message(log, 'M', 1, "Writing sky model binary file: %s", filename);
        oskar_sky_write(filename, sky, status);
    }

    return sky;
}


static void set_up_osm(oskar_Sky* sky, oskar_Log* log,
        const oskar_SettingsSkyOskar* s, double ra0, double dec0, int* status)
{
    int i;
    const char* filename;
    oskar_Sky* t;

    /* Load OSKAR sky model files. */
    if (*status) return;
    for (i = 0; i < s->num_files; ++i)
    {
        filename = s->file[i];
        if (filename && strlen(filename) > 0)
        {
            int binary_file_error = 0;

            /* Load into a temporary sky model. */
            oskar_log_message(log, 'M', 0,
                    "Loading OSKAR sky model file '%s' ...", filename);

            /* Try to read sky model as a binary file first. */
            /* If this fails, read it as an ASCII file. */
            t = oskar_sky_read(filename, OSKAR_CPU, &binary_file_error);
            if (binary_file_error)
                t = oskar_sky_load(filename, oskar_sky_precision(sky), status);

            /* Apply filters and extended source over-ride. */
            set_up_filter(t, &s->filter, ra0, dec0, status);
            set_up_extended(t, &s->extended_sources, status);

            /* Append to sky model. */
            oskar_sky_append(sky, t, status);
            oskar_sky_free(t, status);
            oskar_log_message(log, 'M', 1, "done.");
        }
    }
}


static void set_up_gsm(oskar_Sky* sky, oskar_Log* log,
        const oskar_SettingsSkyGsm* s, double ra0, double dec0, int* status)
{
    int num_pixels, nside;
    const char* filename;
    oskar_Sky* t;
    oskar_Mem* data;

    /* GSM sky model file. */
    filename = s->file;
    if (*status || !filename || strlen(filename) == 0)
        return;

    /* Load the file. */
    oskar_log_message(log, 'M', 0, "Loading GSM data...");
    data = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    num_pixels = (int) oskar_mem_load_ascii(filename, 1, status, data, "");

    /* Compute nside from npix. */
    nside = oskar_healpix_npix_to_nside(num_pixels);
    if (12 * nside * nside != num_pixels)
    {
        oskar_mem_free(data, status);
        *status = OSKAR_ERR_BAD_GSM_FILE;
        return;
    }

    /* Convert brightness temperature to Jy. */
    oskar_convert_brightness_to_jy(data, 0.0, (4.0 * M_PI) / num_pixels,
            s->freq_hz, 0.0, 0.0, "K", "K", 1, status);

    /* Create a temporary sky model. */
    t = oskar_sky_from_healpix_ring(oskar_sky_precision(sky), data,
            s->freq_hz, s->spectral_index, nside, 1, status);
    oskar_mem_free(data, status);

    /* Apply filters and extended source over-ride. */
    set_up_filter(t, &s->filter, ra0, dec0, status);
    set_up_extended(t, &s->extended_sources, status);

    /* Append to sky model. */
    oskar_sky_append(sky, t, status);
    oskar_sky_free(t, status);
    oskar_log_message(log, 'M', 1, "done.");
}


static void set_up_fits_image(oskar_Sky* sky, oskar_Log* log,
        const oskar_SettingsSkyFitsImage* s, double ra0, double dec0,
        int* status)
{
    int i;

    /* Load FITS image files. */
    if (*status) return;
    for (i = 0; i < s->num_files; ++i)
    {
        const char* filename;
        oskar_Sky* t;

        /* Get the filename. */
        filename = s->file[i];
        if (!filename || strlen(filename) == 0)
            continue;

        /* Load the image pixels. */
        oskar_log_message(log, 'M', 0, "Loading FITS file '%s' ...", filename);

        /* Convert the image into a sky model. */
        t = oskar_sky_from_fits_file(oskar_sky_precision(sky), filename,
                s->min_peak_fraction, s->min_abs_val, s->default_map_units,
                s->override_map_units, 0.0, s->spectral_index, status);
        if (*status == OSKAR_ERR_BAD_UNITS)
            oskar_log_error(log,
                    "Units error: Need K, mK, Jy/pixel or "
                    "Jy/beam and beam size.");

        /* Apply filters. */
        set_up_filter(t, &s->filter, ra0, dec0, status);

        /* Append to sky model. */
        oskar_sky_append(sky, t, status);
        oskar_sky_free(t, status);
        if (*status) return;
        oskar_log_message(log, 'M', 1, "done.");
    }
}


static void set_up_healpix_fits(oskar_Sky* sky, oskar_Log* log,
        const oskar_SettingsSkyHealpixFits* s, double ra0, double dec0,
        int* status)
{
    int i;

    /* Load HEALPix FITS image files. */
    if (*status) return;
    for (i = 0; i < s->num_files; ++i)
    {
        oskar_Sky* t;
        const char* filename = s->file[i];

        /* Get the filename. */
        filename = s->file[i];
        if (!filename || strlen(filename) == 0)
            continue;

        /* Read the data from file. */
        oskar_log_message(log, 'M', 0, "Loading HEALPix FITS file '%s' ...",
                filename);

        /* Convert the image into a sky model. */
        t = oskar_sky_from_fits_file(oskar_sky_precision(sky), filename,
                s->min_peak_fraction, s->min_abs_val, s->default_map_units,
                s->override_map_units, s->freq_hz, s->spectral_index, status);
        if (*status == OSKAR_ERR_BAD_UNITS)
            oskar_log_error(log,
                    "Units error: Need K, mK, Jy/pixel or "
                    "Jy/beam and beam size.");

        /* Apply filters and extended source over-ride. */
        set_up_filter(t, &s->filter, ra0, dec0, status);
        set_up_extended(t, &s->extended_sources, status);

        /* Append to sky model. */
        oskar_sky_append(sky, t, status);
        oskar_sky_free(t, status);
        if (*status) return;
        oskar_log_message(log, 'M', 1, "done.");
    }
}


static void set_up_gen_grid(oskar_Sky* sky, oskar_Log* log,
        const oskar_SettingsSkyGeneratorGrid* s, double ra0, double dec0,
        int* status)
{
    oskar_Sky* t;

    /* Check if generator is enabled. */
    if (*status || s->side_length <= 0)
        return;

    /* Generate a sky model containing the grid. */
    oskar_log_message(log, 'M', 0, "Generating source grid positions...");
    t = oskar_sky_generate_grid(oskar_sky_precision(sky), ra0, dec0,
            s->side_length, s->fov_rad, s->mean_flux_jy, s->std_flux_jy,
            s->seed, status);

    /* Apply polarisation and extended source over-ride. */
    set_up_pol(t, &s->pol, status);
    set_up_extended(t, &s->extended_sources, status);

    /* Append to sky model. */
    oskar_sky_append(sky, t, status);
    oskar_sky_free(t, status);
    oskar_log_message(log, 'M', 1, "done.");
}


static void set_up_gen_healpix(oskar_Sky* sky, oskar_Log* log,
        const oskar_SettingsSkyGeneratorHealpix* s, double ra0, double dec0,
        int* status)
{
    int i, nside, npix, type;
    oskar_Sky* t;

    /* Get the HEALPix generator parameters. */
    nside = s->nside;
    if (*status || nside <= 0)
        return;

    /* Generate the new positions into a temporary sky model. */
    npix = 12 * nside * nside;
    type = oskar_sky_precision(sky);
    t = oskar_sky_create(type, OSKAR_CPU, npix, status);
    oskar_log_message(log, 'M', 0, "Generating HEALPix source positions...");
#pragma omp parallel for private(i)
    for (i = 0; i < npix; ++i)
    {
        double ra, dec;
        oskar_convert_healpix_ring_to_theta_phi_d(nside, i, &dec, &ra);
        dec = M_PI / 2.0 - dec;
        oskar_sky_set_source(t, i, ra, dec, s->amplitude, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, status);
    }

    /* Apply filters and extended source over-ride. */
    set_up_filter(t, &s->filter, ra0, dec0, status);
    set_up_extended(t, &s->extended_sources, status);

    /* Append to sky model. */
    oskar_sky_append(sky, t, status);
    oskar_sky_free(t, status);
    oskar_log_message(log, 'M', 1, "done.");
}


static void set_up_gen_rpl(oskar_Sky* sky, oskar_Log* log,
        const oskar_SettingsSkyGeneratorRandomPowerLaw* s, double ra0,
        double dec0, int* status)
{
    oskar_Sky* t;

    /* Random power-law generator. */
    if (*status || s->num_sources <= 0)
        return;

    /* Generate the sources into a temporary sky model. */
    oskar_log_message(log, 'M', 0,
            "Generating random power law source distribution...");
    t = oskar_sky_generate_random_power_law(oskar_sky_precision(sky),
            s->num_sources, s->flux_min, s->flux_max, s->power,
            s->seed, status);

    /* Apply filters and extended source over-ride. */
    set_up_filter(t, &s->filter, ra0, dec0, status);
    set_up_extended(t, &s->extended_sources, status);

    /* Append to sky model. */
    oskar_sky_append(sky, t, status);
    oskar_sky_free(t, status);
    oskar_log_message(log, 'M', 1, "done.");
}


static void set_up_gen_rbpl(oskar_Sky* sky, oskar_Log* log,
        const oskar_SettingsSkyGeneratorRandomBrokenPowerLaw* s, double ra0,
        double dec0, int* status)
{
    int i, num_sources, type;
    oskar_Sky* t;

    /* Random broken power-law generator. */
    num_sources = s->num_sources;
    if (*status || num_sources <= 0)
        return;

    /* Generate the new positions into a temporary sky model. */
    type = oskar_sky_precision(sky);
    t = oskar_sky_create(type, OSKAR_CPU, num_sources, status);
    oskar_log_message(log, 'M', 0,
            "Generating random broken power law source distribution...");

    /* Cannot parallelise here, since rand() is not thread safe. */
    srand(s->seed);
    for (i = 0; i < num_sources; ++i)
    {
        double ra, dec, b;
        oskar_generate_random_coordinate(&ra, &dec);
        b = oskar_random_broken_power_law(s->flux_min, s->flux_max,
                s->threshold, s->power1, s->power2);
        oskar_sky_set_source(t, i, ra, dec, b, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, status);
    }

    /* Apply filters and extended source over-ride. */
    set_up_filter(t, &s->filter, ra0, dec0, status);
    set_up_extended(t, &s->extended_sources, status);

    /* Append to sky model. */
    oskar_sky_append(sky, t, status);
    oskar_sky_free(t, status);
    oskar_log_message(log, 'M', 1, "done.");
}


static void set_up_filter(oskar_Sky* sky, const oskar_SettingsSkyFilter* f,
        double ra0_rad, double dec0_rad, int* status)
{
    oskar_sky_filter_by_flux(sky, f->flux_min, f->flux_max, status);
    oskar_sky_filter_by_radius(sky, f->radius_inner_rad, f->radius_outer_rad,
            ra0_rad, dec0_rad, status);
}


static void set_up_extended(oskar_Sky* sky,
        const oskar_SettingsSkyExtendedSources* ext, int* status)
{
    /* Apply extended source over-ride. */
    if (ext->FWHM_major_rad > 0.0 || ext->FWHM_minor_rad > 0.0)
    {
        oskar_sky_set_gaussian_parameters(sky, ext->FWHM_major_rad,
                ext->FWHM_minor_rad, ext->position_angle_rad, status);
    }
}


static void set_up_pol(oskar_Sky* sky,
        const oskar_SettingsSkyPolarisation* pol, int* status)
{
    oskar_sky_override_polarisation(sky, pol->mean_pol_fraction,
            pol->std_pol_fraction, pol->mean_pol_angle_rad,
            pol->std_pol_angle_rad, pol->seed, status);
}


#ifdef __cplusplus
}
#endif
