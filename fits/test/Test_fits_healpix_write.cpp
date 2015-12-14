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

#include <gtest/gtest.h>
#include <fits/oskar_fits_healpix_write_image.h>
#include <oskar_image.h>

#include <cstdio>
#include <cstdlib>

#if 0
TEST(fits_healpix_image_write, test)
{
    int status = OSKAR_SUCCESS;
    const char* filename = "tmp_test_healpix_image.fits";
    oskar_Image image;
    int nside = 128;
    int npix = 12 * nside * nside;

    int type = OSKAR_DOUBLE;
    int loc = OSKAR_CPU;
    int num_channels = 2;
    oskar_mem_init(&image.data, type, loc, npix * num_channels, 1, &status);
    double* data_ = oskar_mem_double(oskar_image_data(&image), &status);
    for (int c = 0; c < num_channels; ++c)
    {
        for (int i = 0; i < npix; ++i)
        {
            data_[i] =  (double)rand() / ((double)RAND_MAX + 1.0);
            data_[i] *= ((double)c+1.0);
        }
    }

    image.healpix_nside = nside;
    image.num_channels = num_channels;
    oskar_fits_healpix_write_image(filename, &image, &status);
    oskar_mem_free(&image.data, &status);
}
#endif
