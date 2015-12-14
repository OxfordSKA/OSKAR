/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#include <gtest/gtest.h>

#include "fits/oskar_fits_write.h"
#include "fits/oskar_fits_write_axis_header.h"
#include <fitsio.h>
#include <oskar_file_exists.h>
#include <oskar_mem.h>
#include <oskar_cmath.h>
#include <oskar_version.h>

#include <cstdio>
#include <cstdlib>

static double fov_to_cellsize(double fov_deg, int num_pixels)
{
    double max, inc;
    max = sin(fov_deg * M_PI / 360.0); /* Divide by 2. */
    inc = max / (0.5 * num_pixels);
    return asin(inc) * 180.0 / M_PI;
}

static
fitsfile* create_fits_file(const char* filename, int precision, int width,
        int height, int num_times, int num_channels, double centre_deg[2],
        double fov_deg[2], double start_time_mjd, double delta_time_sec,
        double start_freq_hz, double delta_freq_hz, int* status)
{
    int imagetype;
    long naxes[4];
    double delta;
    fitsfile* f = 0;

    /* Create a new FITS file and write the image headers. */
    if (oskar_file_exists(filename)) remove(filename);
    imagetype = (precision == OSKAR_DOUBLE ? DOUBLE_IMG : FLOAT_IMG);
    naxes[0]  = width;
    naxes[1]  = height;
    naxes[2]  = num_times;
    naxes[3]  = num_channels;
    fits_create_file(&f, filename, status);
    fits_create_img(f, imagetype, 4, naxes, status);
    fits_write_date(f, status);
    fits_write_key_str(f, "TELESCOP",
            "OSKAR " OSKAR_VERSION_STR, NULL, status);
    fits_write_history(f, "Created using OSKAR " OSKAR_VERSION_STR, status);

    /* Write axis headers. */
    delta = fov_to_cellsize(fov_deg[0], width);
    oskar_fits_write_axis_header(f, 1, "RA---SIN", "Right Ascension",
            centre_deg[0], -delta, (width + 1) / 2.0, 0.0, status);
    delta = fov_to_cellsize(fov_deg[1], height);
    oskar_fits_write_axis_header(f, 2, "DEC--SIN", "Declination",
            centre_deg[1], delta, (height + 1) / 2.0, 0.0, status);
    oskar_fits_write_axis_header(f, 3, "UTC", "Time",
            start_time_mjd, delta_time_sec, 1.0, 0.0, status);
    oskar_fits_write_axis_header(f, 4, "FREQ", "Frequency",
            start_freq_hz, delta_freq_hz, 1.0, 0.0, status);

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


TEST(fits_write, test)
{
    int status = 0;
    int columns = 10; // width
    int rows = 40; // height
    int planes = 4;
    int blocks = 4;
    int num_elements = columns * rows * planes * blocks;
    oskar_Mem* data = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU,
            num_elements, &status);
    const char filename[] = "temp_test_fits_write.fits";

    // Define test data.
    double* d = oskar_mem_double(data, &status);
    int i = 0;
    for (int b = 0; b < blocks; ++b)
    {
        for (int p = 0; p < planes; ++p)
        {
            if (p == 0)
            {
                for (int r = 0; r < rows; ++r)
                {
                    for (int c = 0; c < columns; ++c)
                    {
                        d[i] = r + 2 * c;
                        ++i;
                    }
                }
            }
            else
            {
                for (int r = 0; r < rows; ++r)
                {
                    for (int c = 0; c < columns; ++c)
                    {
                        d[i] = (b+1) * sin((p+1) * M_PI * r / (double)(rows-1));
                        ++i;
                    }
                }
            }
        }
    }

    long naxes[4];
    double crval[4], crpix[4], cdelt[4], crota[4];

    /* Axis types. */
    const char* ctype[] = {
            "RA---SIN",
            "DEC--SIN",
            "FREQ",
            "STOKES"
    };

    /* Axis comments. */
    const char* ctype_comment[] = {
            "Right Ascension",
            "Declination",
            "Frequency",
            "Polarisation"
    };

    /* Axis dimensions. */
    naxes[0] = columns; // width
    naxes[1] = rows; // height
    naxes[2] = planes;
    naxes[3] = blocks;

    /* Reference values. */
    crval[0] = 10.0; // RA
    crval[1] = 80.0; // DEC
    crval[2] = 100e6;
    crval[3] = 1.0;

    /* Deltas. */
    cdelt[0] = -0.1; // DELTA_RA
    cdelt[1] = 0.1; // DELTA_DEC
    cdelt[2] = 1e5; // BANDWIDTH
    cdelt[3] = 1.0;

    /* Reference pixels. */
    crpix[0] = (columns + 1) / 2.0;
    crpix[1] = (rows + 1) / 2.0;
    crpix[2] = 1.0;
    crpix[3] = 1.0;

    /* Rotation. */
    crota[0] = 0.0;
    crota[1] = 0.0;
    crota[2] = 0.0;
    crota[3] = 0.0;

    /* Write multi-dimensional image data. */
    oskar_fits_write(filename, oskar_mem_type(data), 4, naxes,
            oskar_mem_void(data), ctype, ctype_comment,
            crval, cdelt, crpix, crota, &status);
    oskar_mem_free(data, &status);
}
