/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_brightness_to_jy.h"
#include "log/oskar_log.h"
#include "math/oskar_cmath.h"
#include "sky/oskar_sky.h"
#include <fitsio.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Sky* oskar_sky_from_fits_file(int precision, const char* filename,
        double min_peak_fraction, double min_abs_val,
        const char* default_map_units, int override_units, double frequency_hz,
        double spectral_index, int* status)
{
    double image_crval_deg[2], image_crpix[2];
    double image_cellsize_deg = 0.0, image_freq_hz = 0.0;
    double beam_area_pixels = 0.0, pixel_area_sr = 0.0;
    char *reported_map_units = 0, ordering = 0, coordsys = 0;
    int naxis = 0, nside = 0;
    int image_size[2];
    oskar_Sky* t = 0;
    oskar_Mem* data = 0;
    fitsfile* fptr = 0;

    /* Determine whether this is a regular FITS image or HEALPix data. */
    fits_open_file(&fptr, filename, READONLY, status);
    if (*status || !fptr)
    {
        *status = OSKAR_ERR_FILE_IO;
        return 0;
    }
    fits_get_img_dim(fptr, &naxis, status);
    fits_close_file(fptr, status);
    if (naxis == 0)
    {
        /* Try to load HEALPix data. */
        data = oskar_mem_read_healpix_fits(filename, 0,
                &nside, &ordering, &coordsys, &reported_map_units, status);
        if (data)
        {
            pixel_area_sr = (4.0 * M_PI) / oskar_mem_length(data);
        }

        /* Check HEALPix ordering scheme. */
        if (!*status && ordering != 'R')
        {
            *status = OSKAR_ERR_FILE_IO;
            oskar_log_error(0, "HEALPix data is not in RING format.");
        }
    }
    else
    {
        /* Try to load image pixels. */
        data = oskar_mem_read_fits_image_plane(filename, 0, 0, 0,
                image_size, image_crval_deg, image_crpix,
                &image_cellsize_deg, 0, &image_freq_hz, &beam_area_pixels,
                &reported_map_units, status);
        pixel_area_sr = pow(image_cellsize_deg * M_PI / 180.0, 2.0);
    }

    /* Make sure pixels are in Jy. */
    if (image_freq_hz == 0.0)
    {
        image_freq_hz = frequency_hz;
    }
    oskar_convert_brightness_to_jy(data, beam_area_pixels, pixel_area_sr,
            image_freq_hz, min_peak_fraction, min_abs_val, reported_map_units,
            default_map_units, override_units, status);

    /* Convert the image into a sky model. */
    if (naxis == 0)
    {
        t = oskar_sky_from_healpix_ring(precision, data, image_freq_hz,
                spectral_index, nside, (coordsys == 'G'), status);
    }
    else
    {
        t = oskar_sky_from_image(precision, data,
                image_size, image_crval_deg, image_crpix,
                image_cellsize_deg, image_freq_hz, spectral_index, status);
    }

    /* Free pixel data and return sky model. */
    free(reported_map_units);
    oskar_mem_free(data, status);
    return t;
}

#ifdef __cplusplus
}
#endif
