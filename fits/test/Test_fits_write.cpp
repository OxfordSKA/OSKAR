/*
 * Copyright (c) 2012-2013, The University of Oxford
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
#include <oskar_mem.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

TEST(fits_write, test)
{
    int status = 0;
    int columns = 10; // width
    int rows = 40; // height
    int planes = 4;
    int blocks = 4;
    int num_elements = columns * rows * planes * blocks;
    oskar_Mem data(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_elements);
    const char filename[] = "temp_test_fits_write.fits";

    // Define test data.
    double* d = oskar_mem_double(&data, &status);
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
    oskar_fits_write(filename, oskar_mem_type(&data), 4, naxes,
            oskar_mem_void(&data), ctype, ctype_comment,
            crval, cdelt, crpix, crota, &status);
}
