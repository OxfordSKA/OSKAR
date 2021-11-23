/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include <fitsio.h>
#include "convert/oskar_convert_fov_to_cellsize.h"
#include "mem/oskar_mem.h"
#include "math/oskar_cmath.h"

#include <cstdio>
#include <cstdlib>

static void write_axis(fitsfile* fptr, int axis_id, const char* ctype,
        double crval, double cdelt, double crpix, int* status)
{
    char key[FLEN_KEYWORD], value[FLEN_VALUE];
    int decimals = 10;
    strncpy(value, ctype, FLEN_VALUE-1);
    fits_make_keyn("CTYPE", axis_id, key, status);
    fits_write_key_str(fptr, key, value, NULL, status);
    fits_make_keyn("CRVAL", axis_id, key, status);
    fits_write_key_dbl(fptr, key, crval, decimals, NULL, status);
    fits_make_keyn("CDELT", axis_id, key, status);
    fits_write_key_dbl(fptr, key, cdelt, decimals, NULL, status);
    fits_make_keyn("CRPIX", axis_id, key, status);
    fits_write_key_dbl(fptr, key, crpix, decimals, NULL, status);
    fits_make_keyn("CROTA", axis_id, key, status);
    fits_write_key_dbl(fptr, key, 0.0, decimals, NULL, status);
}


static fitsfile* create_fits_file(const char* fname, int precision, int width,
        int height, int num_times, int num_channels, double centre_deg[2],
        double fov_deg[2], double start_time_mjd, double delta_time_sec,
        double start_freq_hz, double delta_freq_hz, int* status)
{
    int imagetype = 0;
    long naxes[4];
    double delta = 0.0;
    fitsfile* f = 0;
    FILE* file = 0;

    /* If the file exists, remove it. */
    if ((file = fopen(fname, "r")) != NULL)
    {
        fclose(file);
        remove(fname);
    }

    /* Create a new FITS file and write the image headers. */
    imagetype = (precision == OSKAR_DOUBLE ? DOUBLE_IMG : FLOAT_IMG);
    naxes[0]  = width;
    naxes[1]  = height;
    naxes[2]  = num_times;
    naxes[3]  = num_channels;
    fits_create_file(&f, fname, status);
    fits_create_img(f, imagetype, 4, naxes, status);
    fits_write_date(f, status);

    /* Write axis headers. */
    delta = oskar_convert_fov_to_cellsize(fov_deg[0] * M_PI/180.0, width);
    write_axis(f, 1, "RA---SIN",
            centre_deg[0], -delta, (width + 1) / 2.0, status);
    delta = oskar_convert_fov_to_cellsize(fov_deg[1] * M_PI/180.0, height);
    write_axis(f, 2, "DEC--SIN",
            centre_deg[1], delta, (height + 1) / 2.0, status);
    write_axis(f, 3, "UTC", start_time_mjd, delta_time_sec, 1.0, status);
    write_axis(f, 4, "FREQ", start_freq_hz, delta_freq_hz, 1.0, status);

    /* Write other headers. */
    fits_write_key_str(f, "TIMESYS", "UTC", NULL, status);
    fits_write_key_str(f, "TIMEUNIT", "s", "Time axis units", status);
    fits_write_key_dbl(f, "MJD-OBS", start_time_mjd, 10, "Start time", status);
    fits_write_key_dbl(f, "OBSRA", centre_deg[0], 10, "RA", status);
    fits_write_key_dbl(f, "OBSDEC", centre_deg[1], 10, "DEC", status);

    return f;
}


TEST(fits_write, test_planes)
{
    int status = 0;
    int width = 10;
    int height = 40;
    int num_times = 4;
    int num_channels = 4;
    int precision = OSKAR_DOUBLE;
    oskar_Mem* data = oskar_mem_create(precision, OSKAR_CPU,
            width * height, &status);
    const char filename[] = "temp_test_fits_write_planes.fits";

    // Create the FITS file.
    double phase_centre_deg[2] = {10., 80.};
    double fov_deg[2] = {1., 4.};
    fitsfile* f = create_fits_file(filename, precision, width, height,
            num_times, num_channels, phase_centre_deg, fov_deg,
            1.0, 1.0, 100e6, 1e5, &status);
    int datatype = (precision == OSKAR_DOUBLE) ? TDOUBLE : TFLOAT;

    // Define test data.
    double* d = oskar_mem_double(data, &status);
    for (int c = 0; c < num_channels; ++c)
    {
        for (int t = 0; t < num_times; ++t)
        {
            int i = 0; // Pixel index within plane.
            if (t == 0)
            {
                for (int h = 0; h < height; ++h)
                {
                    for (int w = 0; w < width; ++w)
                    {
                        d[i++] = h + 2 * w;
                    }
                }
            }
            else
            {
                for (int h = 0; h < height; ++h)
                {
                    for (int w = 0; w < width; ++w)
                    {
                        d[i++] = (c+1) *
                                sin((t+1) * M_PI * h / (double)(height-1));
                    }
                }
            }

            // Write plane to FITS file.
            long firstpix[] = {1, 1, t + 1, c + 1};
            fits_write_pix(f, datatype, firstpix, width * height, d, &status);
        }
    }

    // Close the FITS file.
    fits_close_file(f, &status);
    oskar_mem_free(data, &status);
}
