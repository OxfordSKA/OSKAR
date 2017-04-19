/*
 * Copyright (c) 2016-2017, The University of Oxford
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

#include "sky/oskar_sky.h"
#include "convert/oskar_convert_relative_directions_to_lon_lat.h"
#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif

static void set_pixel(oskar_Sky* sky, int i, int x, int y, double val,
        const double crval[2], const double crpix[2], const double cdelt[2],
        double image_freq_hz, double spectral_index, int* status);


oskar_Sky* oskar_sky_from_image(int precision, const oskar_Mem* image,
        const int image_size[2], const double image_crval_deg[2],
        const double image_crpix[2], double image_cellsize_deg,
        double image_freq_hz, double spectral_index, int* status)
{
    int i, type, x, y;
    double crval[2], cdelt[2], val;
    oskar_Sky* sky;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Get reference pixels and reference values in radians. */
    crval[0] = image_crval_deg[0] * M_PI / 180.0;
    crval[1] = image_crval_deg[1] * M_PI / 180.0;

    /* Compute sine of pixel deltas for inverse orthographic projection. */
    cdelt[0] = -sin(image_cellsize_deg * M_PI / 180.0);
    cdelt[1] = -cdelt[0];

    /* Create a sky model. */
    sky = oskar_sky_create(precision, OSKAR_CPU, 0, status);

    /* Store the image pixels. */
    type = oskar_mem_precision(image);
    if (type == OSKAR_SINGLE)
    {
        const float *img = oskar_mem_float_const(image, status);
        for (y = 0, i = 0; y < image_size[1]; ++y)
        {
            for (x = 0; x < image_size[0]; ++x)
            {
                /* Check pixel value. */
                val = (double) (img[image_size[0] * y + x]);
                if (val == 0.0)
                    continue;

                set_pixel(sky, i++, x, y, val, crval, image_crpix, cdelt,
                        image_freq_hz, spectral_index, status);
            }
        }
    }
    else
    {
        const double *img = oskar_mem_double_const(image, status);
        for (y = 0, i = 0; y < image_size[1]; ++y)
        {
            for (x = 0; x < image_size[0]; ++x)
            {
                /* Check pixel value. */
                val = img[image_size[0] * y + x];
                if (val == 0.0)
                    continue;

                set_pixel(sky, i++, x, y, val, crval, image_crpix, cdelt,
                        image_freq_hz, spectral_index, status);
            }
        }
    }

    /* Return the sky model. */
    oskar_sky_resize(sky, i, status);
    return sky;
}


static void set_pixel(oskar_Sky* sky, int i, int x, int y, double val,
        const double crval[2], const double crpix[2], const double cdelt[2],
        double image_freq_hz, double spectral_index, int* status)
{
    double ra, dec, l, m;

    /* Convert pixel positions to RA and Dec values. */
    l = cdelt[0] * (x + 1 - crpix[0]);
    m = cdelt[1] * (y + 1 - crpix[1]);
    oskar_convert_relative_directions_to_lon_lat_2d_d(1,
            &l, &m, crval[0], crval[1], &ra, &dec);

    /* Store pixel data in sky model. */
    if (oskar_sky_num_sources(sky) <= i)
        oskar_sky_resize(sky, i + 1000, status);
    oskar_sky_set_source(sky, i, ra, dec, val, 0.0, 0.0, 0.0,
            image_freq_hz, spectral_index, 0.0, 0.0, 0.0, 0.0, status);
}

#ifdef __cplusplus
}
#endif
