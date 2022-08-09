/*
 * Copyright (c) 2012-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "beam_pattern/oskar_beam_pattern.h"
#include "beam_pattern/private_beam_pattern.h"
#include "beam_pattern/private_beam_pattern_generate_coordinates.h"
#include "convert/oskar_convert_fov_to_cellsize.h"
#include "math/oskar_cmath.h"
#include "math/private_cond2_2x2.h"
#include "utility/oskar_device.h"
#include "utility/oskar_file_exists.h"
#include "oskar_version.h"

#include <stdlib.h>
#include <string.h>

#if __STDC_VERSION__ >= 199901L
#define SNPRINTF(BUF, SIZE, FMT, ...) snprintf(BUF, SIZE, FMT, __VA_ARGS__);
#else
#define SNPRINTF(BUF, SIZE, FMT, ...) sprintf(BUF, FMT, __VA_ARGS__);
#endif

#ifdef __cplusplus
extern "C" {
#endif


static void set_up_host_data(oskar_BeamPattern* h, int *status);
static void create_averaged_products(oskar_BeamPattern* h, int ta, int ca,
        int* status);
static void set_up_device_data(oskar_BeamPattern* h, int* status);
static void write_axis(fitsfile* fptr, int axis_id, const char* ctype,
        const char* ctype_comment, double crval, double cdelt, double crpix,
        int* status);
static fitsfile* create_fits_file(const char* filename, int precision,
        int width, int height, int num_times, int num_channels,
        const double centre_deg[2], const double fov_deg[2],
        double start_time_mjd, double delta_time_sec,
        double start_freq_hz, double delta_freq_hz,
        int horizon_mode, const char* settings_log, size_t settings_log_length,
        int* status);
static int data_product_index(oskar_BeamPattern* h, int data_product_type,
        int stokes_in, int stokes_out, int i_station, int time_average,
        int channel_average);
static char* construct_filename(oskar_BeamPattern* h, int data_product_type,
        int stokes_in, int stokes_out, int i_station, int time_average,
        int channel_average, const char* ext);
static void new_fits_file(oskar_BeamPattern* h, int data_product_type,
        int stokes_in, int stokes_out, int i_station, int channel_average,
        int time_average, int* status);
static void new_text_file(oskar_BeamPattern* h, int data_product_type,
        int stokes_in, int stokes_out, int i_station, int channel_average,
        int time_average, int* status);
static const char* data_type_to_string(int type);
static const char* stokes_type_to_string(int type);


void oskar_beam_pattern_check_init(oskar_BeamPattern* h, int* status)
{
    if (*status) return;

    /* Check that the telescope model has been set. */
    if (!h->tel)
    {
        oskar_log_error(h->log, "Telescope model not set.");
        *status = OSKAR_ERR_SETTINGS_TELESCOPE;
        return;
    }

    /* Check that each compute device has been set up. */
    set_up_host_data(h, status);
    set_up_device_data(h, status);
}


static void set_up_host_data(oskar_BeamPattern* h, int *status)
{
    int i = 0, k = 0;
    size_t j = 0;
    if (*status) return;

    /* Set up pixel positions. */
    oskar_beam_pattern_generate_coordinates(h,
            OSKAR_COORDS_RADEC, status);

    /* Work out how many pixel chunks have to be processed. */
    h->num_chunks = (h->num_pixels + h->max_chunk_size - 1) / h->max_chunk_size;

    /* Create scratch arrays for output pixel data. */
    if (!h->pix)
    {
        h->pix = oskar_mem_create(h->prec, OSKAR_CPU,
                h->max_chunk_size, status);
        h->ctemp = oskar_mem_create(h->prec | OSKAR_COMPLEX, OSKAR_CPU,
                h->max_chunk_size, status);
    }

    /* Get the contents of the log at this point so we can write a
     * reasonable file header. Replace newlines with zeros. */
    h->settings_log_length = 0;
    free(h->settings_log);
    h->settings_log = oskar_log_file_data(h->log, &h->settings_log_length);
    for (j = 0; j < h->settings_log_length; ++j)
    {
        if (h->settings_log[j] == '\n') h->settings_log[j] = 0;
        if (h->settings_log[j] == '\r') h->settings_log[j] = ' ';
    }

    /* Return if data products already exist. */
    if (h->data_products) return;

    /* Create a file for each requested data product. */
    /* Voltage amplitude and phase can only be generated if there is
     * no averaging. */
    if (h->separate_time_and_channel)
    {
        /* Create station-level data products. */
        for (i = 0; i < h->num_active_stations; ++i)
        {
            /* Text file. */
            if (h->voltage_raw_txt)
            {
                new_text_file(h, RAW_COMPLEX, -1, -1, i, 0, 0, status);
            }
            if (h->voltage_amp_txt)
            {
                if (h->pol_mode == OSKAR_POL_MODE_SCALAR)
                {
                    new_text_file(h, AMP, -1, -1, i, 0, 0, status);
                }
                else
                {
                    for (k = XX; k <= YY; ++k)
                    {
                        new_text_file(h, AMP, -1, k, i, 0, 0, status);
                    }
                }
            }
            if (h->voltage_phase_txt)
            {
                if (h->pol_mode == OSKAR_POL_MODE_SCALAR)
                {
                    new_text_file(h, PHASE, -1, -1, i, 0, 0, status);
                }
                else
                {
                    for (k = XX; k <= YY; ++k)
                    {
                        new_text_file(h, PHASE, -1, k, i, 0, 0, status);
                    }
                }
            }
            if (h->ixr_txt && h->pol_mode == OSKAR_POL_MODE_FULL)
            {
                new_text_file(h, IXR, -1, -1, i, 0, 0, status);
            }

            /* Can only create images if coordinates are on a grid. */
            if (h->coord_grid_type != 'B') continue;

            /* FITS file. */
            if (h->voltage_amp_fits)
            {
                if (h->pol_mode == OSKAR_POL_MODE_SCALAR)
                {
                    new_fits_file(h, AMP, -1, -1, i, 0, 0, status);
                }
                else
                {
                    for (k = XX; k <= YY; ++k)
                    {
                        new_fits_file(h, AMP, -1, k, i, 0, 0, status);
                    }
                }
            }
            if (h->voltage_phase_fits)
            {
                if (h->pol_mode == OSKAR_POL_MODE_SCALAR)
                {
                    new_fits_file(h, PHASE, -1, -1, i, 0, 0, status);
                }
                else
                {
                    for (k = XX; k <= YY; ++k)
                    {
                        new_fits_file(h, PHASE, -1, k, i, 0, 0, status);
                    }
                }
            }
            if (h->ixr_fits && h->pol_mode == OSKAR_POL_MODE_FULL)
            {
                new_fits_file(h, IXR, -1, -1, i, 0, 0, status);
            }
        }
    }

    /* Create data products that can be averaged. */
    if (h->separate_time_and_channel)
    {
        create_averaged_products(h, 0, 0, status);
    }
    if (h->average_time_and_channel)
    {
        create_averaged_products(h, 1, 1, status);
    }
    if (h->average_single_axis == 'C')
    {
        create_averaged_products(h, 0, 1, status);
    }
    else if (h->average_single_axis == 'T')
    {
        create_averaged_products(h, 1, 0, status);
    }

    /* Check that at least one output file will be generated. */
    if (h->num_data_products == 0 && !*status)
    {
        *status = OSKAR_ERR_FILE_IO;
        oskar_log_error(h->log, "No output file(s) selected.");
    }
}


static void create_averaged_products(oskar_BeamPattern* h, int ta, int ca,
        int* status)
{
    int s = 0, i = 0, o = 0;
    if (*status) return;

    /* Create station-level data products that can be averaged. */
    for (s = 0; s < h->num_active_stations; ++s)
    {
        /* Text file. */
        for (i = 0; i < 2; ++i)
        {
            for (o = I; (o <= V) && h->stokes[i] && h->auto_power_txt; ++o)
            {
                new_text_file(h, AUTO_POWER_AMP, i, o, s, ta, ca, status);
            }
        }

        /* Can only create images if coordinates are on a grid. */
        if (h->coord_grid_type != 'B') continue;

        /* FITS file. */
        for (i = 0; i < 2; ++i)
        {
            for (o = I; (o <= V) && h->stokes[i]; ++o)
            {
                if (h->auto_power_fits)
                {
                    new_fits_file(h, AUTO_POWER_AMP, i, o, s, ta, ca, status);
                }
                if (h->auto_power_phase_fits)
                {
                    new_fits_file(h, AUTO_POWER_PHASE, i, o, s, ta, ca, status);
                }
                if (h->auto_power_real_fits)
                {
                    new_fits_file(h, AUTO_POWER_REAL, i, o, s, ta, ca, status);
                }
                if (h->auto_power_imag_fits)
                {
                    new_fits_file(h, AUTO_POWER_IMAG, i, o, s, ta, ca, status);
                }
            }
        }
    }

    /* Text file. */
    for (i = 0; i < 2; ++i)
    {
        if (h->cross_power_raw_txt && h->stokes[i])
        {
            new_text_file(h, CROSS_POWER_RAW_COMPLEX, i, -1, -1, ta, ca,
                    status);
        }
        for (o = I; (o <= V) && h->stokes[i]; ++o)
        {
            if (h->cross_power_amp_txt)
            {
                new_text_file(h, CROSS_POWER_AMP, i, o, -1, ta, ca, status);
            }
            if (h->cross_power_phase_txt)
            {
                new_text_file(h, CROSS_POWER_PHASE, i, o, -1, ta, ca, status);
            }
        }
    }

    /* Can only create images if coordinates are on a grid. */
    if (h->coord_grid_type != 'B') return;

    /* FITS file. */
    for (i = 0; i < 2; ++i)
    {
        for (o = I; (o <= V) && h->stokes[i]; ++o)
        {
            if (h->cross_power_amp_fits)
            {
                new_fits_file(h, CROSS_POWER_AMP, i, o, -1, ta, ca, status);
            }
            if (h->cross_power_phase_fits)
            {
                new_fits_file(h, CROSS_POWER_PHASE, i, o, -1, ta, ca, status);
            }
            if (h->cross_power_real_fits)
            {
                new_fits_file(h, CROSS_POWER_REAL, i, o, -1, ta, ca, status);
            }
            if (h->cross_power_imag_fits)
            {
                new_fits_file(h, CROSS_POWER_IMAG, i, o, -1, ta, ca, status);
            }
        }
    }
}


static void write_axis(fitsfile* fptr, int axis_id, const char* ctype,
        const char* ctype_comment, double crval, double cdelt, double crpix,
        int* status)
{
    char key[FLEN_KEYWORD], value[FLEN_VALUE], comment[FLEN_COMMENT];
    int decimals = 10;
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
    fits_write_key_dbl(fptr, key, 0.0, decimals, NULL, status);
}


static fitsfile* create_fits_file(const char* filename, int precision,
        int width, int height, int num_times, int num_channels,
        const double centre_deg[2], const double fov_deg[2],
        double start_time_mjd, double delta_time_sec,
        double start_freq_hz, double delta_freq_hz,
        int horizon_mode, const char* settings_log, size_t settings_log_length,
        int* status)
{
    int imagetype = 0;
    long naxes[4], naxes_dummy[4] = {1L, 1L, 1L, 1L};
    double delta = 0.0;
    const double deg2rad = M_PI / 180.0;
    const double rad2deg = 180.0 / M_PI;
    fitsfile* f = 0;
    const char* line = 0;
    size_t length = 0;
    if (*status) return 0;

    /* Create a new FITS file and write the image headers. */
    if (oskar_file_exists(filename)) remove(filename);
    imagetype = (precision == OSKAR_DOUBLE ? DOUBLE_IMG : FLOAT_IMG);
    naxes[0]  = width;
    naxes[1]  = height;
    naxes[2]  = num_channels;
    naxes[3]  = num_times;
    fits_create_file(&f, filename, status);
    fits_create_img(f, imagetype, 4, naxes_dummy, status);
    fits_write_date(f, status);
    fits_write_key_str(f, "TELESCOP", "OSKAR " OSKAR_VERSION_STR, 0, status);

    /* Write axis headers. */
    if (horizon_mode)
    {
        fits_write_key_str(f, "WCSNAME", "AZELGEO", NULL, status);
        fits_write_key_dbl(f, "LONPOLE", 0.0, 10, NULL, status);
        fits_write_key_dbl(f, "LATPOLE", 90.0, 10, NULL, status);
        delta = oskar_convert_fov_to_cellsize(M_PI, width);
        write_axis(f, 1, "HOLN-SIN", "Azimuth",
                0.0, delta * rad2deg, (width + 1) / 2.0, status);
        delta = oskar_convert_fov_to_cellsize(M_PI, height);
        write_axis(f, 2, "HOLT-SIN", "Elevation",
                90.0, delta * rad2deg, (height + 1) / 2.0, status);
    }
    else
    {
        delta = oskar_convert_fov_to_cellsize(fov_deg[0] * deg2rad, width);
        write_axis(f, 1, "RA---SIN", "Right Ascension",
                centre_deg[0], -delta * rad2deg, (width + 1) / 2.0, status);
        delta = oskar_convert_fov_to_cellsize(fov_deg[1] * deg2rad, height);
        write_axis(f, 2, "DEC--SIN", "Declination",
                centre_deg[1], delta * rad2deg, (height + 1) / 2.0, status);
    }
    write_axis(f, 3, "FREQ", "Frequency",
            start_freq_hz, delta_freq_hz, 1.0, status);
    write_axis(f, 4, "UTC", "Time",
            start_time_mjd * 86400.0, delta_time_sec, 1.0, status);

    /* Write other headers. */
    fits_write_key_str(f, "TIMESYS", "UTC", NULL, status);
    fits_write_key_str(f, "TIMEUNIT", "s", "Time axis units", status);
    fits_write_key_dbl(f, "MJD-OBS", start_time_mjd, 10, "Start time", status);
    if (!horizon_mode)
    {
        fits_write_key_dbl(f, "OBSRA", centre_deg[0], 10, "RA", status);
        fits_write_key_dbl(f, "OBSDEC", centre_deg[1], 10, "DEC", status);
    }

    /* Write the settings log up to this point as HISTORY comments. */
    line = settings_log;
    length = settings_log_length;
    for (; settings_log_length > 0;)
    {
        const char* eol = 0;
        fits_write_history(f, line, status);
        eol = (const char*) memchr(line, '\0', length);
        if (!eol) break;
        eol += 1;
        length -= (eol - line);
        line = eol;
    }

    /* Update header keywords with the correct axis lengths.
     * Needs to be done here because CFITSIO doesn't let us write only the
     * file header with the correct axis lengths to start with. This trick
     * allows us to create a small dummy image block to write only the headers,
     * and not waste effort moving a huge block of zeros within the file. */
    fits_update_key_lng(f, "NAXIS1", naxes[0], 0, status);
    fits_update_key_lng(f, "NAXIS2", naxes[1], 0, status);
    fits_update_key_lng(f, "NAXIS3", naxes[2], 0, status);
    fits_update_key_lng(f, "NAXIS4", naxes[3], 0, status);

    return f;
}


static int data_product_index(oskar_BeamPattern* h, int data_product_type,
        int stokes_in, int stokes_out, int i_station, int time_average,
        int channel_average)
{
    int i = 0;
    for (i = 0; (i < h->num_data_products) && h->data_products; ++i)
    {
        if (h->data_products[i].type == data_product_type &&
                h->data_products[i].stokes_in == stokes_in &&
                h->data_products[i].stokes_out == stokes_out &&
                h->data_products[i].i_station == i_station &&
                h->data_products[i].time_average == time_average &&
                h->data_products[i].channel_average == channel_average)
        {
            break;
        }
    }
    if (i == h->num_data_products)
    {
        i = h->num_data_products++;
        h->data_products = (DataProduct*) realloc(h->data_products,
                h->num_data_products * sizeof(DataProduct));
        memset(&(h->data_products[i]), 0, sizeof(DataProduct));
        h->data_products[i].type = data_product_type;
        h->data_products[i].stokes_in = stokes_in;
        h->data_products[i].stokes_out = stokes_out;
        h->data_products[i].i_station = i_station;
        h->data_products[i].time_average = time_average;
        h->data_products[i].channel_average = channel_average;
    }
    return i;
}


static char* construct_filename(oskar_BeamPattern* h, int data_product_type,
        int stokes_in, int stokes_out, int i_station, int time_average,
        int channel_average, const char* ext)
{
    int buflen = 0, start = 0;
    char* name = 0;

    /* Construct the filename. */
    buflen = (int) strlen(h->root_path) + 100;
    name = (char*) calloc(buflen, 1);
    start += SNPRINTF(name + start, buflen - start, "%s", h->root_path);
    if (i_station >= 0)
        start += SNPRINTF(name + start, buflen - start, "_S%04d",
                h->station_ids[i_station]);
    start += SNPRINTF(name + start, buflen - start, "_%s",
            time_average ? "TIME_AVG" : "TIME_SEP");
    start += SNPRINTF(name + start, buflen - start, "_%s",
            channel_average ? "CHAN_AVG" : "CHAN_SEP");
    start += SNPRINTF(name + start, buflen - start, "_%s",
            data_type_to_string(data_product_type));
    if (stokes_in == 0)
        start += SNPRINTF(name + start, buflen - start, "_%s", "I");
    if (stokes_in == 1)
        start += SNPRINTF(name + start, buflen - start, "_%s", "CUSTOM");
    if (stokes_out >= 0)
        start += SNPRINTF(name + start, buflen - start, "_%s",
                stokes_type_to_string(stokes_out));
    (void) SNPRINTF(name + start, buflen - start, ".%s", ext);
    return name;
}


static void new_fits_file(oskar_BeamPattern* h, int data_product_type,
        int stokes_in, int stokes_out, int i_station, int time_average,
        int channel_average, int* status)
{
    int i = 0, horizon_mode = 0;
    char* name = 0;
    fitsfile* f = 0;
    if (*status) return;

    /* Check polarisation type is possible. */
    if ((stokes_in > I || stokes_out > I) && h->pol_mode != OSKAR_POL_MODE_FULL)
    {
        return;
    }

    /* Construct the filename. */
    name = construct_filename(h, data_product_type, stokes_in, stokes_out,
            i_station, time_average, channel_average, "fits");

    /* Open the file. */
    horizon_mode = h->coord_frame_type == 'H';
    f = create_fits_file(name, h->prec, h->width, h->height,
            (time_average ? 1 : h->num_time_steps),
            (channel_average ? 1 : h->num_channels),
            h->phase_centre_deg, h->fov_deg, h->time_start_mjd_utc,
            h->time_inc_sec, h->freq_start_hz, h->freq_inc_hz,
            horizon_mode, h->settings_log, h->settings_log_length, status);
    if (!f || *status)
    {
        *status = OSKAR_ERR_FILE_IO;
        free(name);
        return;
    }
    i = data_product_index(h, data_product_type, stokes_in, stokes_out,
            i_station, time_average, channel_average);
    if (h->data_products) h->data_products[i].fits_file = f;
    free(name);
}


static void new_text_file(oskar_BeamPattern* h, int data_product_type,
        int stokes_in, int stokes_out, int i_station, int time_average,
        int channel_average, int* status)
{
    int i = 0;
    char* name = 0;
    FILE* f = 0;
    if (*status) return;

    /* Check polarisation type is possible. */
    if ((stokes_in > 0 || stokes_out > I) && h->pol_mode != OSKAR_POL_MODE_FULL)
    {
        return;
    }

    /* Construct the filename. */
    name = construct_filename(h, data_product_type, stokes_in, stokes_out,
            i_station, time_average, channel_average, "txt");

    /* Open the file. */
    f = fopen(name, "w");
    if (!f)
    {
        *status = OSKAR_ERR_FILE_IO;
        free(name);
        return;
    }
    if (i_station >= 0)
    {
        fprintf(f, "# Beam pixel list for station %d\n",
                h->station_ids[i_station]);
    }
    else
    {
        fprintf(f, "# Beam pixel list for telescope (interferometer)\n");
    }
    fprintf(f, "# Filename is '%s'\n", name);
    fprintf(f, "# Dimension order (slowest to fastest) is:\n");
    if (h->average_single_axis != 'T')
    {
        fprintf(f, "#     [pixel chunk], [time], [channel], [pixel index]\n");
    }
    else
    {
        fprintf(f, "#     [pixel chunk], [channel], [time], [pixel index]\n");
    }
    fprintf(f, "# Number of pixel chunks: %d\n", h->num_chunks);
    fprintf(f, "# Number of times (output): %d\n",
            time_average ? 1 : h->num_time_steps);
    fprintf(f, "# Number of channels (output): %d\n",
            channel_average ? 1 : h->num_channels);
    fprintf(f, "# Maximum pixel chunk size: %d\n", h->max_chunk_size);
    fprintf(f, "# Total number of pixels: %d\n", h->num_pixels);
    i = data_product_index(h, data_product_type, stokes_in, stokes_out,
            i_station, time_average, channel_average);
    if (h->data_products) h->data_products[i].text_file = f;
    free(name);
}


static const char* data_type_to_string(int type)
{
    switch (type)
    {
    case RAW_COMPLEX:                return "RAW_COMPLEX";
    case AMP:                        return "AMP";
    case PHASE:                      return "PHASE";
    case AUTO_POWER_AMP:             return "AUTO_POWER_AMP";
    case AUTO_POWER_PHASE:           return "AUTO_POWER_PHASE";
    case AUTO_POWER_REAL:            return "AUTO_POWER_REAL";
    case AUTO_POWER_IMAG:            return "AUTO_POWER_IMAG";
    case CROSS_POWER_RAW_COMPLEX:    return "CROSS_POWER_RAW_COMPLEX";
    case CROSS_POWER_AMP:            return "CROSS_POWER_AMP";
    case CROSS_POWER_PHASE:          return "CROSS_POWER_PHASE";
    case CROSS_POWER_REAL:           return "CROSS_POWER_REAL";
    case CROSS_POWER_IMAG:           return "CROSS_POWER_IMAG";
    case IXR:                        return "IXR";
    default:                         return "";
    }
}


static const char* stokes_type_to_string(int type)
{
    switch (type)
    {
    case I:  return "I";
    case Q:  return "Q";
    case U:  return "U";
    case V:  return "V";
    case XX: return "XX";
    case XY: return "XY";
    case YX: return "YX";
    case YY: return "YY";
    default: return "";
    }
}


static void set_up_device_data(oskar_BeamPattern* h, int* status)
{
    int i = 0, beam_type = 0, max_src = 0, max_size = 0;
    int auto_power = 0, cross_power = 0, raw_data = 0;
    if (*status) return;

    /* Get local variables. */
    max_src = h->max_chunk_size;
    max_size = h->num_active_stations * max_src;
    beam_type = h->prec | OSKAR_COMPLEX;
    if (h->pol_mode == OSKAR_POL_MODE_FULL)
    {
        beam_type |= OSKAR_MATRIX;
    }
    raw_data = h->ixr_txt || h->ixr_fits ||
            h->voltage_raw_txt || h->voltage_amp_txt || h->voltage_phase_txt ||
            h->voltage_amp_fits || h->voltage_phase_fits;
    auto_power = h->auto_power_txt || h->auto_power_fits ||
            h->auto_power_phase_fits ||
            h->auto_power_real_fits || h->auto_power_imag_fits;
    cross_power = h->cross_power_raw_txt ||
            h->cross_power_amp_fits || h->cross_power_phase_fits ||
            h->cross_power_amp_txt || h->cross_power_phase_txt ||
            h->cross_power_real_fits || h->cross_power_imag_fits;

    /* Expand the number of devices to the number of selected GPUs,
     * if required. */
    if (h->num_devices < h->num_gpus)
    {
        oskar_beam_pattern_set_num_devices(h, h->num_gpus);
    }

    for (i = 0; i < h->num_devices; ++i)
    {
        int dev_loc = 0, i_stokes_type = 0;
        DeviceData* d = &h->d[i];
        if (*status) break;

        /* Select the device. */
        if (i < h->num_gpus)
        {
            oskar_device_set(h->dev_loc, h->gpu_ids[i], status);
            dev_loc = h->dev_loc;
        }
        else
        {
            dev_loc = OSKAR_CPU;
        }

        /* Device memory. */
        d->previous_chunk_index = -1;
        if (!d->tel)
        {
            d->jones_data = oskar_mem_create(beam_type, dev_loc, max_size,
                    status);
            d->jones_temp = oskar_mem_create(beam_type, dev_loc, 0, status);
            d->lon_rad = oskar_mem_create(h->prec, dev_loc, 1 + max_src, status);
            d->lat_rad = oskar_mem_create(h->prec, dev_loc, 1 + max_src, status);
            d->x    = oskar_mem_create(h->prec, dev_loc, 1 + max_src, status);
            d->y    = oskar_mem_create(h->prec, dev_loc, 1 + max_src, status);
            d->z    = oskar_mem_create(h->prec, dev_loc, 1 + max_src, status);
            d->tel  = oskar_telescope_create_copy(h->tel, dev_loc, status);
            d->work = oskar_station_work_create(h->prec, dev_loc, status);
            oskar_station_work_set_isoplanatic_screen(d->work,
                    oskar_telescope_isoplanatic_screen(d->tel));
            oskar_station_work_set_tec_screen_common_params(d->work,
                    oskar_telescope_ionosphere_screen_type(d->tel),
                    oskar_telescope_tec_screen_height_km(d->tel),
                    oskar_telescope_tec_screen_pixel_size_m(d->tel),
                    oskar_telescope_tec_screen_time_interval_sec(d->tel));
            if (oskar_telescope_ionosphere_screen_type(d->tel) == 'E')
            {
                oskar_station_work_set_tec_screen_path(d->work,
                        oskar_telescope_tec_screen_path(d->tel));
            }
        }

        /* Host memory. */
        if (!d->jones_data_cpu[0] && raw_data)
        {
            d->jones_data_cpu[0] = oskar_mem_create(beam_type, OSKAR_CPU,
                    max_size, status);
            d->jones_data_cpu[1] = oskar_mem_create(beam_type, OSKAR_CPU,
                    max_size, status);
        }

        /* Auto-correlation beam output arrays. */
        for (i_stokes_type = 0; i_stokes_type < 2; ++i_stokes_type)
        {
            if (!h->stokes[i_stokes_type]) continue;

            if (!d->auto_power[i_stokes_type] && auto_power)
            {
                /* Device memory. */
                d->auto_power[i_stokes_type] = oskar_mem_create(
                        beam_type, dev_loc, max_size, status);
                oskar_mem_clear_contents(d->auto_power[i_stokes_type], status);

                /* Host memory. */
                d->auto_power_cpu[i_stokes_type][0] = oskar_mem_create(
                        beam_type, OSKAR_CPU, max_size, status);
                d->auto_power_cpu[i_stokes_type][1] = oskar_mem_create(
                        beam_type, OSKAR_CPU, max_size, status);
                if (h->average_single_axis == 'T')
                {
                    d->auto_power_time_avg[i_stokes_type] = oskar_mem_create(
                            beam_type, OSKAR_CPU, max_size, status);
                }
                if (h->average_single_axis == 'C')
                {
                    d->auto_power_channel_avg[i_stokes_type] = oskar_mem_create(
                            beam_type, OSKAR_CPU, max_size, status);
                }
                if (h->average_time_and_channel)
                {
                    d->auto_power_channel_and_time_avg[i_stokes_type] =
                            oskar_mem_create(beam_type, OSKAR_CPU,
                                    max_size, status);
                }
            }

            /* Cross-correlation beam output arrays. */
            if (!d->cross_power[i_stokes_type] && cross_power)
            {
                if (h->num_active_stations < 2)
                {
                    oskar_log_error(h->log, "Cannot create cross-power beam "
                            "using less than two active stations.");
                    *status = OSKAR_ERR_INVALID_ARGUMENT;
                    break;
                }

                /* Device memory. */
                d->cross_power[i_stokes_type] = oskar_mem_create(
                        beam_type, dev_loc, max_src, status);
                oskar_mem_clear_contents(d->cross_power[i_stokes_type], status);

                /* Host memory. */
                d->cross_power_cpu[i_stokes_type][0] = oskar_mem_create(
                        beam_type, OSKAR_CPU, max_src, status);
                d->cross_power_cpu[i_stokes_type][1] = oskar_mem_create(
                        beam_type, OSKAR_CPU, max_src, status);
                if (h->average_single_axis == 'T')
                {
                    d->cross_power_time_avg[i_stokes_type] = oskar_mem_create(
                            beam_type, OSKAR_CPU, max_src, status);
                }
                if (h->average_single_axis == 'C')
                {
                    d->cross_power_channel_avg[i_stokes_type] = oskar_mem_create(
                            beam_type, OSKAR_CPU, max_src, status);
                }
                if (h->average_time_and_channel)
                {
                    d->cross_power_channel_and_time_avg[i_stokes_type] =
                            oskar_mem_create(beam_type, OSKAR_CPU,
                                    max_src, status);
                }
            }
        }

        /* Timers. */
        if (!d->tmr_compute)
        {
            d->tmr_compute = oskar_timer_create(OSKAR_TIMER_NATIVE);
        }
    }
}

#ifdef __cplusplus
}
#endif
