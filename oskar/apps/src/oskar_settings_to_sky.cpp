/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "apps/oskar_settings_to_sky.h"

#include "convert/oskar_convert_brightness_to_jy.h"
#include "convert/oskar_convert_healpix_ring_to_theta_phi.h"
#include "math/oskar_healpix_npix_to_nside.h"
#include "math/oskar_random_broken_power_law.h"
#include "math/oskar_random_gaussian.h"
#include "sky/oskar_generate_random_coordinate.h"
#include "sky/oskar_sky.h"
#include "utility/oskar_get_error_string.h"

#include "math/oskar_cmath.h"
#include <cstdlib> /* For srand() */
#include <cstring>

using oskar::SettingsTree;

#define DEG2RAD (M_PI / 180.0)
#define ARCSEC2RAD (M_PI / 648000.0)

static void load_osm(oskar_Sky* sky, SettingsTree* s,
        double ra0, double dec0, oskar_Log* log, int* status);
#if 0
static void load_gsm(oskar_Sky* sky, SettingsTree* s,
        double ra0, double dec0, oskar_Log* log, int* status);
#endif
static void load_fits_image(oskar_Sky* sky, SettingsTree* s,
        double ra0, double dec0, oskar_Log* log, int* status);
static void load_healpix_fits(oskar_Sky* sky, SettingsTree* s,
        double ra0, double dec0, oskar_Log* log, int* status);

static void gen_grid(oskar_Sky* sky, SettingsTree* s,
        double ra0, double dec0, oskar_Log* log, int* status);
static void gen_healpix(oskar_Sky* sky, SettingsTree* s,
        double ra0, double dec0, oskar_Log* log, int* status);
static void gen_rpl(oskar_Sky* sky, SettingsTree* s,
        double ra0, double dec0, oskar_Log* log, int* status);
static void gen_rbpl(oskar_Sky* sky, SettingsTree* s,
        double ra0, double dec0, oskar_Log* log, int* status);

static void set_up_filter(oskar_Sky* sky, SettingsTree* s,
        double ra0_rad, double dec0_rad, oskar_Log* log, int* status);
static void set_up_extended(oskar_Sky* sky, SettingsTree* s, int* status);

oskar_Sky* oskar_settings_to_sky(SettingsTree* s, oskar_Log* log, int* status)
{
    const char* filename = 0;
    if (*status || !s) return 0;
    s->clear_group();

    /* Create an empty sky model. */
    oskar_log_section(log, 'M', "Sky model set-up");
    const int type = s->to_int("simulator/double_precision", status) ?
            OSKAR_DOUBLE : OSKAR_SINGLE;
    oskar_Sky* sky = oskar_sky_create(type, OSKAR_CPU, 0, status);
    s->begin_group("observation");
    double ra0  = s->to_double("phase_centre_ra_deg", status) * DEG2RAD;
    double dec0 = s->to_double("phase_centre_dec_deg", status) * DEG2RAD;

    /* Disable filters and grid generator in drift scan mode. */
    if (s->starts_with("mode", "Drift", status))
    {
        dec0 = -100.0;
    }

    s->end_group();
    s->begin_group("sky");

    /* Load sky model data files. */
    load_osm(sky, s, ra0, dec0, log, status);
    //load_gsm(sky, s, ra0, dec0, log, status);
    load_fits_image(sky, s, ra0, dec0, log, status);
    load_healpix_fits(sky, s, ra0, dec0, log, status);

    /* Generate sky models from generator parameters. */
    gen_grid(sky, s, ra0, dec0, log, status);
    gen_healpix(sky, s, ra0, dec0, log, status);
    gen_rpl(sky, s, ra0, dec0, log, status);
    gen_rbpl(sky, s, ra0, dec0, log, status);

    /* Return if sky model contains no sources. */
    int num_sources = oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES);
    if (num_sources == 0)
    {
        oskar_log_warning(log, "Sky model contains no sources.");
        s->clear_group();
        return sky;
    }

    /* Perform final overrides. */
    if (s->to_int("spectral_index/override", status))
    {
        double mean = s->to_double("spectral_index/mean", status);
        double std_dev = s->to_double("spectral_index/std_dev", status);
        double ref = s->to_double("spectral_index/ref_frequency_hz", status);
        int seed = s->to_int("spectral_index/seed", status);
        oskar_log_message(
                log, 'M', 0, "Overriding source spectral index values..."
        );
        oskar_mem_set_value_real(
                oskar_sky_column(sky, OSKAR_SKY_REF_HZ, 0, status),
                ref, 0, num_sources, status
        );
        for (int i = 0; i < num_sources; ++i)
        {
            double val[2];
            oskar_random_gaussian2(seed, i, 0, val);
            val[0] = std_dev * val[0] + mean;
            oskar_mem_set_element_real(
                    oskar_sky_column(sky, OSKAR_SKY_SPEC_IDX, 0, status),
                    i, val[0], status
            );
        }
        oskar_log_message(log, 'M', 1, "done.");
    }

    if (*status)
    {
        s->clear_group();
        return sky;
    }

    /* Write text file. */
    filename = s->to_string("output_text_file", status);
    if (filename && strlen(filename) > 0 && !*status)
    {
        oskar_log_message(log, 'M', 1,
                "Writing sky model text file: %s", filename);
        const int use_named_columns = s->to_int(
                "output_text_file/use_named_columns", status
        );
        if (use_named_columns)
        {
            const int use_degrees = s->to_int(
                    "output_text_file/use_degrees", status
            );
            const int write_name = s->to_int(
                    "output_text_file/write_name", status
            );
            const int write_type = s->to_int(
                    "output_text_file/write_type", status
            );
            oskar_sky_save_named_columns(
                    sky, filename, use_degrees, write_name, write_type, status
            );
        }
        else
        {
            oskar_sky_save(sky, filename, status);
        }
    }

    s->clear_group();
    return sky;
}


static void load_osm(oskar_Sky* sky, SettingsTree* s,
        double ra0, double dec0, oskar_Log* log, int* status)
{
    int num_files = 0;
    s->begin_group("oskar_sky_model");
    const char* const* files = s->to_string_list("file", &num_files, status);
    for (int i = 0; i < num_files; ++i)
    {
        if (*status) break;
        if (!files[i] || strlen(files[i]) == 0) continue;

        /* Load into a temporary sky model. */
        oskar_log_message(
                log, 'M', 0, "Loading sky model file '%s' ...", files[i]
        );
        oskar_Sky* t = oskar_sky_load(
                files[i], oskar_sky_int(sky, OSKAR_SKY_PRECISION), status
        );

        /* Apply filters and extended source over-ride. */
        set_up_filter(t, s, ra0, dec0, log, status);
        set_up_extended(t, s, status);

        /* Append to sky model. */
        if (!*status)
        {
            oskar_sky_append(sky, t, status);
            oskar_log_message(log, 'M', 1, "done.");
        }
        oskar_sky_free(t, status);
    }
    s->end_group();
}

#if 0
static void load_gsm(oskar_Sky* sky, SettingsTree* s,
        double ra0, double dec0, oskar_Log* log, int* status)
{
    const char* filename = s->to_string("gsm/file", status);
    if (*status || !filename || strlen(filename) == 0) return;

    /* Load the file. */
    oskar_log_message(log, 'M', 0, "Loading GSM data...");
    oskar_Mem* data = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    int num_pixels = (int) oskar_mem_load_ascii(filename,
            1, status, data, "");

    /* Compute nside from npix. */
    int nside = oskar_healpix_npix_to_nside(num_pixels);
    if (12 * nside * nside != num_pixels)
    {
        oskar_mem_free(data, status);
        *status = OSKAR_ERR_BAD_GSM_FILE;
        return;
    }

    /* Convert brightness temperature to Jy. */
    s->begin_group("gsm");
    double freq_hz = s->to_double("freq_hz", status);
    double spix = s->to_double("spectral_index", status);
    oskar_convert_brightness_to_jy(data, 0.0, (4.0 * M_PI) / num_pixels,
            freq_hz, 0.0, 0.0, "K", "K", 1, status);

    /* Create a temporary sky model. */
    oskar_Sky* t = oskar_sky_from_healpix_ring(
            oskar_sky_int(sky, OSKAR_SKY_PRECISION),
            data, freq_hz, spix, nside, 1, status
    );
    oskar_mem_free(data, status);

    /* Apply filters and extended source over-ride. */
    set_up_filter(t, s, ra0, dec0, status);
    set_up_extended(t, s, status);
    s->end_group();

    /* Append to sky model. */
    if (!*status)
    {
        oskar_sky_append(sky, t, status);
        oskar_log_message(log, 'M', 1, "done.");
    }
    oskar_sky_free(t, status);
}
#endif

static void load_fits_image(oskar_Sky* sky, SettingsTree* s,
        double ra0, double dec0, oskar_Log* log, int* status)
{
    int num_files = 0;
    s->begin_group("fits_image");
    const char* const* files = s->to_string_list("file", &num_files, status);
    const char* default_map_units = s->to_string("default_map_units", status);
    int override_map_units = s->to_int("override_map_units", status);
    double min_peak_fraction = s->to_double("min_peak_fraction", status);
    double min_abs_val = s->to_double("min_abs_val", status);
    double spectral_index = s->to_double("spectral_index", status);
    for (int i = 0; i < num_files; ++i)
    {
        if (*status) break;
        if (!files[i] || strlen(files[i]) == 0) continue;
        oskar_log_message(log, 'M', 0, "Loading FITS file '%s' ...", files[i]);

        /* Convert the image into a sky model. */
        oskar_Sky* t = oskar_sky_from_fits_file(
                oskar_sky_int(sky, OSKAR_SKY_PRECISION),
                files[i], min_peak_fraction, min_abs_val,
                default_map_units, override_map_units,
                0.0, spectral_index, status
        );
        if (*status == OSKAR_ERR_BAD_UNITS)
        {
            oskar_log_error(log, "Units error: Need K, mK, Jy/pixel or "
                    "Jy/beam and beam size.");
        }

        /* Apply filters. */
        set_up_filter(t, s, ra0, dec0, log, status);

        /* Append to sky model. */
        if (!*status)
        {
            oskar_sky_append(sky, t, status);
            oskar_log_message(log, 'M', 1, "done.");
        }
        oskar_sky_free(t, status);
    }
    s->end_group();
}


static void load_healpix_fits(oskar_Sky* sky, SettingsTree* s,
        double ra0, double dec0, oskar_Log* log, int* status)
{
    int num_files = 0;
    s->begin_group("healpix_fits");
    const char* const* files = s->to_string_list("file", &num_files, status);
    const char* default_map_units = s->to_string("default_map_units", status);
    int override_map_units = s->to_int("override_map_units", status);
    double min_peak_fraction = s->to_double("min_peak_fraction", status);
    double min_abs_val = s->to_double("min_abs_val", status);
    double spectral_index = s->to_double("spectral_index", status);
    double freq_hz = s->to_double("freq_hz", status);
    for (int i = 0; i < num_files; ++i)
    {
        if (*status) break;
        if (!files[i] || strlen(files[i]) == 0) continue;

        /* Read the data from file. */
        oskar_log_message(log, 'M', 0,
                "Loading HEALPix FITS file '%s' ...", files[i]);

        /* Convert the image into a sky model. */
        oskar_Sky* t = oskar_sky_from_fits_file(
                oskar_sky_int(sky, OSKAR_SKY_PRECISION),
                files[i], min_peak_fraction, min_abs_val,
                default_map_units, override_map_units,
                freq_hz, spectral_index, status
        );
        if (*status == OSKAR_ERR_BAD_UNITS)
        {
            oskar_log_error(log, "Units error: Need K, mK, Jy/pixel or "
                    "Jy/beam and beam size.");
        }

        /* Apply filters and extended source over-ride. */
        set_up_filter(t, s, ra0, dec0, log, status);
        set_up_extended(t, s, status);

        /* Append to sky model. */
        if (!*status)
        {
            oskar_sky_append(sky, t, status);
            oskar_log_message(log, 'M', 1, "done.");
        }
        oskar_sky_free(t, status);
    }
    s->end_group();
}


static void gen_grid(oskar_Sky* sky, SettingsTree* s,
        double ra0, double dec0, oskar_Log* log, int* status)
{
    /* Check if generator is enabled. */
    int side_length = s->to_int("generator/grid/side_length", status);
    if (*status || side_length <= 0)
    {
        return;
    }
    if (dec0 < -99.0)
    {
        oskar_log_warning(log,
                "Cannot use sky model grid generator in drift-scan mode.");
        return;
    }

    /* Generate a sky model containing the grid. */
    oskar_log_message(log, 'M', 0, "Generating source grid positions...");
    s->begin_group("generator/grid");
    double fov_rad = s->to_double("fov_deg", status) * DEG2RAD;
    double mean_flux_jy = s->to_double("mean_flux_jy", status);
    double std_flux_jy = s->to_double("std_flux_jy", status);
    int seed = s->to_int("seed", status);
    oskar_Sky* t = oskar_sky_generate_grid(
            oskar_sky_int(sky, OSKAR_SKY_PRECISION), ra0, dec0,
            side_length, fov_rad, mean_flux_jy, std_flux_jy, seed, status
    );

    /* Apply extended source over-ride. */
    set_up_extended(t, s, status);
    s->end_group();

    /* Append to sky model. */
    if (!*status)
    {
        oskar_sky_append(sky, t, status);
        oskar_log_message(log, 'M', 1, "done.");
    }
    oskar_sky_free(t, status);
}


static void gen_healpix(oskar_Sky* sky, SettingsTree* s,
        double ra0, double dec0, oskar_Log* log, int* status)
{
    /* Check if generator is enabled. */
    int nside = s->to_int("generator/healpix/nside", status);
    if (*status || nside <= 0)
    {
        return;
    }

    /* Generate the new positions into a temporary sky model. */
    oskar_log_message(log, 'M', 0, "Generating HEALPix source positions...");
    int npix = 12 * nside * nside;
    int type = oskar_sky_int(sky, OSKAR_SKY_PRECISION);
    oskar_Sky* t = oskar_sky_create(type, OSKAR_CPU, npix, status);
    s->begin_group("generator/healpix");
    double amplitude = s->to_double("amplitude", status);
    for (int i = 0; i < npix; ++i)
    {
        double ra = 0.0, dec = 0.0;
        oskar_convert_healpix_ring_to_theta_phi_pixel(nside, i, &dec, &ra);
        dec = M_PI / 2.0 - dec;
        oskar_sky_set_source(t, i, ra, dec, amplitude,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, status);
    }

    /* Apply filters and extended source over-ride. */
    set_up_filter(t, s, ra0, dec0, log, status);
    set_up_extended(t, s, status);
    s->end_group();

    /* Append to sky model. */
    if (!*status)
    {
        oskar_sky_append(sky, t, status);
        oskar_log_message(log, 'M', 1, "done.");
    }
    oskar_sky_free(t, status);
}


static void gen_rpl(oskar_Sky* sky, SettingsTree* s,
        double ra0, double dec0, oskar_Log* log, int* status)
{
    /* Check if generator is enabled. */
    int num_sources = s->to_int(
            "generator/random_power_law/num_sources", status);
    if (*status || num_sources <= 0)
    {
        return;
    }

    /* Generate the sources into a temporary sky model. */
    oskar_log_message(log, 'M', 0,
            "Generating random power law source distribution...");
    s->begin_group("generator/random_power_law");
    double flux_min = s->to_double("flux_min", status);
    double flux_max = s->to_double("flux_max", status);
    double power = s->to_double("power", status);
    int seed = s->to_int("seed", status);
    oskar_Sky* t = oskar_sky_generate_random_power_law(
            oskar_sky_int(sky, OSKAR_SKY_PRECISION),
            num_sources, flux_min, flux_max, power, seed, status
    );

    /* Apply filters and extended source over-ride. */
    set_up_filter(t, s, ra0, dec0, log, status);
    set_up_extended(t, s, status);
    s->end_group();

    /* Append to sky model. */
    if (!*status)
    {
        oskar_sky_append(sky, t, status);
        oskar_log_message(log, 'M', 1, "done.");
    }
    oskar_sky_free(t, status);
}


static void gen_rbpl(oskar_Sky* sky, SettingsTree* s,
        double ra0, double dec0, oskar_Log* log, int* status)
{
    /* Check if generator is enabled. */
    int num_sources = s->to_int(
            "generator/random_broken_power_law/num_sources", status);
    if (*status || num_sources <= 0)
    {
        return;
    }

    /* Generate the sources into a temporary sky model. */
    oskar_log_message(log, 'M', 0,
            "Generating random broken power law source distribution...");
    s->begin_group("generator/random_broken_power_law");
    double flux_min = s->to_double("flux_min", status);
    double flux_max = s->to_double("flux_max", status);
    double threshold = s->to_double("threshold", status);
    double power1 = s->to_double("power1", status);
    double power2 = s->to_double("power2", status);
    int seed = s->to_int("seed", status);
    int type = oskar_sky_int(sky, OSKAR_SKY_PRECISION);
    oskar_Sky* t = oskar_sky_create(type, OSKAR_CPU, num_sources, status);

    /* Cannot parallelise here, since rand() is not thread safe. */
    srand(seed);
    for (int i = 0; i < num_sources; ++i)
    {
        double ra = 0.0, dec = 0.0, b = 0.0;
        oskar_generate_random_coordinate(&ra, &dec);
        b = oskar_random_broken_power_law(flux_min, flux_max,
                threshold, power1, power2);
        oskar_sky_set_source(t, i, ra, dec, b, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, status);
    }

    /* Apply filters and extended source over-ride. */
    set_up_filter(t, s, ra0, dec0, log, status);
    set_up_extended(t, s, status);
    s->end_group();

    /* Append to sky model. */
    if (!*status)
    {
        oskar_sky_append(sky, t, status);
        oskar_log_message(log, 'M', 1, "done.");
    }
    oskar_sky_free(t, status);
}


static void set_up_filter(oskar_Sky* sky, SettingsTree* s,
        double ra0_rad, double dec0_rad, oskar_Log* log, int* status)
{
    s->begin_group("filter");
    double flux_min = s->to_double("flux_min", status);
    double flux_max = s->to_double("flux_max", status);
    double radius_inner_deg = s->to_double("radius_inner_deg", status);
    double radius_outer_deg = s->to_double("radius_outer_deg", status);
    oskar_sky_filter_by_flux(sky, flux_min, flux_max, status);
    if (radius_inner_deg != 0.0 || radius_outer_deg < 180.0)
    {
        if (dec0_rad < -99.0)
        {
            oskar_log_warning(log, "Cannot filter sky model by radius "
                    "from phase centre in drift-scan mode.");
        }
        else
        {
            oskar_sky_filter_by_radius(
                    sky, radius_inner_deg * DEG2RAD, radius_outer_deg * DEG2RAD,
                    ra0_rad, dec0_rad, status
            );
        }
    }
    s->end_group();
}


static void set_up_extended(oskar_Sky* sky, SettingsTree* s, int* status)
{
    s->begin_group("extended_sources");
    const double major_rad = s->to_double("FWHM_major", status) * ARCSEC2RAD;
    const double minor_rad = s->to_double("FWHM_minor", status) * ARCSEC2RAD;
    const double pa_rad = s->to_double("position_angle", status) * DEG2RAD;
    if (major_rad > 0.0 && minor_rad > 0.0)
    {
        const int num_sources = oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES);
        oskar_mem_set_value_real(
                oskar_sky_column(sky, OSKAR_SKY_MAJOR_RAD, 0, status),
                major_rad, 0, num_sources, status
        );
        oskar_mem_set_value_real(
                oskar_sky_column(sky, OSKAR_SKY_MINOR_RAD, 0, status),
                minor_rad, 0, num_sources, status
        );
        oskar_mem_set_value_real(
                oskar_sky_column(sky, OSKAR_SKY_PA_RAD, 0, status),
                pa_rad, 0, num_sources, status
        );
    }
    s->end_group();
}
