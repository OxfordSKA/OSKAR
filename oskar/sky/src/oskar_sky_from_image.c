/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_relative_directions_to_lon_lat.h"
#include "log/oskar_log.h"
#include "math/oskar_cmath.h"
#include "sky/oskar_sky.h"

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
    int i = 0, type = 0, x = 0, y = 0;
    double crval[2], cdelt[2], val = 0.0;
    oskar_Sky* sky = 0;
    if (*status) return 0;

    /* Check pixel size has been defined. */
    if (image_cellsize_deg == 0.0)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        oskar_log_error(0, "Unknown image pixel size. "
                "(Ensure all WCS headers are present.)");
        return 0;
    }

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
                if (val == 0.0) continue;

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
                if (val == 0.0) continue;

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
    double ra = 0.0, dec = 0.0;

    /* Convert pixel positions to RA and Dec values. */
    const double l = cdelt[0] * (x + 1 - crpix[0]);
    const double m = cdelt[1] * (y + 1 - crpix[1]);
    const double cos_dec0 = cos(crval[1]);
    const double sin_dec0 = sin(crval[1]);
    oskar_convert_relative_directions_to_lon_lat_2d_d(1,
            &l, &m, 0, crval[0], cos_dec0, sin_dec0, &ra, &dec);

    /* Store pixel data in sky model. */
    if (oskar_sky_num_sources(sky) <= i)
    {
        oskar_sky_resize(sky, i + 1000, status);
    }
    oskar_sky_set_source(sky, i, ra, dec, val, 0.0, 0.0, 0.0,
            image_freq_hz, spectral_index, 0.0, 0.0, 0.0, 0.0, status);
}

#ifdef __cplusplus
}
#endif
