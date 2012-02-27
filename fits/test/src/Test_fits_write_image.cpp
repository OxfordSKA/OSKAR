/*
 * Copyright (c) 2011, The University of Oxford
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

#include "fits/test/Test_fits_write_image.h"
#include "fits/oskar_fits_write_image.h"
#include "utility/oskar_Mem.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>

void Test_fits_write_image::test_method()
{
    int columns = 10; // width
    int rows = 20; // height
    double ra0 = 10.0;
    double dec0 = 80.0;
    double ra_d = -0.1;
    double dec_d = 0.1;
    double freq = 100e6;
    double bw = 1e5;
    oskar_Mem data(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, columns * rows);
    const char filename[] = "cpp_unit_test_image.fits";

    // Define test data.
    double* d = (double*) data.data;
    for (int r = 0, i = 0; r < rows; ++r)
    {
        for (int c = 0; c < columns; ++c, ++i)
        {
            d[i] = r + 2 * c;
        }
    }

    oskar_fits_write_image(filename, data.type(), columns, rows, data.data,
            ra0, dec0, ra_d, dec_d, freq, bw);
}
