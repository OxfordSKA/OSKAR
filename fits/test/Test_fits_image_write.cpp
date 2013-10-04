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

#include "fits/oskar_fits_image_write.h"
#include "imaging/oskar_image_free.h"
#include "imaging/oskar_image_init.h"
#include "imaging/oskar_image_resize.h"
#include <oskar_mem.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

TEST(fits_image_write, test)
{
    int columns = 10; // width
    int rows = 20; // height
    int err = 0;

    // Create the image.
    oskar_Image image(OSKAR_DOUBLE, OSKAR_LOCATION_CPU);
    oskar_image_resize(&image, columns, rows, 1, 1, 1, &err);
    ASSERT_EQ(0, err);

    // Add image meta-data.
    image.centre_ra_deg = 10.0;
    image.centre_dec_deg = 80.0;
    image.fov_ra_deg = 1.0;
    image.fov_dec_deg = 2.0;
    image.freq_start_hz = 100e6;
    image.freq_inc_hz = 1e5;

    // Define test data.
    double* d = oskar_mem_double(&image.data, &err);
    ASSERT_EQ(0, err);
    for (int r = 0, i = 0; r < rows; ++r)
    {
        for (int c = 0; c < columns; ++c, ++i)
        {
            d[i] = r + 2 * c;
        }
    }

    // Write the data.
    const char filename[] = "temp_test_image.fits";
    oskar_fits_image_write(&image, NULL, filename, &err);

    // Free memory.
    oskar_image_free(&image, &err);
}
