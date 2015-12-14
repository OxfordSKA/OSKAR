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

#include "fits/oskar_fits_image_to_sky_model.h"
#include "fits/oskar_fits_check_status.h"
#include <oskar_convert_relative_directions_to_lon_lat.h>
#include <oskar_log.h>
#include <oskar_sky.h>
#include <oskar_cmath.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fitsio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_AXES 10
#define FACTOR (2.0*sqrt(2.0*log(2.0)))

int oskar_fits_image_to_sky_model(oskar_Log* ptr, const char* filename,
        oskar_Sky* sky, double spectral_index, double min_peak_fraction,
        double noise_floor, int downsample_factor)
{
    int err = 0, i = 0, j = 0, num_pix = 0, status = 0, status2 = 0;
    int naxis = 0, type = 0, imagetype = 0, anynul = 0, jy_beam = 0;
    int x = 0, y = 0, xx = 0, yy = 0, ix = 0, iy = 0, width = 0, height = 0;
    fitsfile* fptr = NULL;
    long naxes[MAX_AXES];
    char card[FLEN_CARD], *ctype[MAX_AXES], ctype_str[MAX_AXES][FLEN_VALUE];
    double crval[MAX_AXES], crpix[MAX_AXES], cdelt[MAX_AXES];
    double nul = 0.0, peak = 0.0, ref_freq = 0.0, val = 0.0, val_new = 0.0;
    double bmaj = 0.0, bmin = 0.0, barea = 0.0, barea_inv = 0.0;
    void* data = 0;
    oskar_Sky* temp_sky = 0;

    /* Check inputs. */
    if (filename == NULL || sky == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;
    if (downsample_factor < 1) downsample_factor = 1;

    /* Open the FITS file. */
    fits_open_file(&fptr, filename, READONLY, &status);
    oskar_fits_check_status(ptr, status, "Opening file");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Get the image parameters. */
    fits_get_img_param(fptr, MAX_AXES, &imagetype, &naxis, naxes, &status);
    oskar_fits_check_status(ptr, status, "Reading image parameters");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Set the data type. */
    if (imagetype != FLOAT_IMG && imagetype != DOUBLE_IMG)
    {
        oskar_log_error(ptr, "Unknown FITS data type.");
        return OSKAR_ERR_FITS_IO;
    }
    type = (imagetype == FLOAT_IMG) ? TFLOAT : TDOUBLE;

    /* Check that the FITS image contains at least two dimensions. */
    if (naxis < 2 || naxis > MAX_AXES)
    {
        oskar_log_error(ptr, "This is not a recognised FITS image.");
        return OSKAR_ERR_FITS_IO;
    }

    /* Read all CTYPE values. */
    for (i = 0; i < MAX_AXES; ++i)
        ctype[i] = ctype_str[i];
    fits_read_keys_str(fptr, "CTYPE", 1, naxis, ctype, &i, &status);
    oskar_fits_check_status(ptr, status, "Reading CTYPE");
    if (status || i == 0) return OSKAR_ERR_FITS_IO;

    /* Check CTYPE1 and CTYPE2. */
    if (strncmp(ctype[0], "RA---SIN", 8) != 0)
    {
        oskar_log_error(ptr, "Unknown CTYPE1 (must be RA---SIN).");
        return OSKAR_ERR_FITS_IO;
    }
    if (strncmp(ctype[1], "DEC--SIN", 8) != 0)
    {
        oskar_log_error(ptr, "Unknown CTYPE2 (must be DEC--SIN).");
        return OSKAR_ERR_FITS_IO;
    }

    /* Read all CDELT values. */
    fits_read_keys_dbl(fptr, "CDELT", 1, naxis, cdelt, &i, &status);
    oskar_fits_check_status(ptr, status, "Reading CDELT");
    if (status || i == 0) return OSKAR_ERR_FITS_IO;
    if (fabs(fabs(cdelt[0]) - fabs(cdelt[1])) > 1e-5)
    {
        oskar_log_error(ptr, "Map pixels are not square.");
        return OSKAR_ERR_FITS_IO;
    }

    /* Read all CRPIX values. */
    fits_read_keys_dbl(fptr, "CRPIX", 1, naxis, crpix, &i, &status);
    oskar_fits_check_status(ptr, status, "Reading CRPIX");
    if (status || i == 0) return OSKAR_ERR_FITS_IO;

    /* Read all CRVAL values. */
    fits_read_keys_dbl(fptr, "CRVAL", 1, naxis, crval, &i, &status);
    oskar_fits_check_status(ptr, status, "Reading CRVAL");
    if (status || i == 0) return OSKAR_ERR_FITS_IO;

    /* Check if there are multiple image planes. */
    for (i = 0, num_pix = 1; i < naxis; ++i)
        num_pix *= naxes[i];
    if (num_pix > naxes[0] * naxes[1])
    {
        num_pix = naxes[0] * naxes[1];
        oskar_log_warning(ptr, "FITS cube contains more than two dimensions. "
                "(Reading only the first plane.)");
    }

    /* Read and check map units. */
    fits_read_key(fptr, TSTRING, "BUNIT", card, NULL, &status);
    oskar_fits_check_status(ptr, status, "Reading BUNIT");
    if (status)
    {
        oskar_log_error(ptr, "Could not determine map units.");
        return OSKAR_ERR_FITS_IO;
    }
    if (strncmp(card, "JY/BEAM", 7) == 0)
        jy_beam = 1;
    else
    {
        if (strncmp(card, "JY/PIXEL", 8))
        {
            oskar_log_error(ptr, "Unknown units: need JY/BEAM or JY/PIXEL");
            return OSKAR_ERR_FITS_IO;
        }
    }

    /* Get the reference frequency. */
    for (i = 0; i < naxis; ++i)
    {
        if (strncmp(ctype[i], "FREQ", 4) == 0)
        {
            ref_freq = crval[i];
            oskar_log_message(ptr, 'M', 0, "Reference frequency is %.3f MHz.",
                    ref_freq / 1e6);
            break;
        }
    }

    /* Search for beam size in header keywords first. */
    fits_read_key(fptr, TDOUBLE, "BMAJ", &bmaj, NULL, &status);
    fits_read_key(fptr, TDOUBLE, "BMIN", &bmin, NULL, &status2);
    if (status || status2)
    {
        int cards = 0;
        status = 0; status2 = 0;

        /* If header keywords don't exist, search all the history cards. */
        fits_get_hdrspace(fptr, &cards, NULL, &status);
        oskar_fits_check_status(ptr, status, "Determining header size");
        if (status) return OSKAR_ERR_FITS_IO;
        for (i = 0; i < cards; ++i)
        {
            fits_read_record(fptr, i, card, &status);
            if (!strncmp(card, "HISTORY AIPS   CLEAN BMAJ", 25))
            {
                sscanf(card + 26, "%lf", &bmaj);
                sscanf(card + 44, "%lf", &bmin);
                break;
            }
        }
    }

    /* Check if we have beam size information. */
    if (bmaj > 0.0 && bmin > 0.0)
    {
        oskar_log_message(ptr, 'M', 0, "Found beam size to be "
                "%.3f x %.3f arcsec.", bmaj * 3600.0, bmin * 3600.0);

        /* Calculate the beam area in pixels (normalisation factor). */
        barea = 2.0 * M_PI * (bmaj * bmin)
                    / (FACTOR * FACTOR * cdelt[0] * cdelt[0]);
    }

    /* Record the beam area. */
    if (barea > 0.0)
        oskar_log_message(ptr, 'M', 0, "Beam area is %.3f pixels.", barea);
    else if (jy_beam)
    {
        oskar_log_error(ptr, "Unknown beam size, and map units are JY/BEAM.");
        return OSKAR_ERR_FITS_IO;
    }

    /* Allocate memory for image data. */
    data = malloc((type == TFLOAT ? sizeof(float) : sizeof(double)) * num_pix);
    if (data == NULL)
    {
        err = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        goto cleanup;
    }

    /* Read image data. */
    fits_read_img(fptr, type, 1, num_pix, &nul, data, &anynul, &status);
    oskar_fits_check_status(ptr, status, "Reading image data");
    if (status) goto cleanup;

    /* Create a temporary sky model. */
    temp_sky = oskar_sky_create(oskar_sky_precision(sky),
            OSKAR_CPU, 0, &err);

    /* Divide pixel values by beam area if required, blank any below noise
     * floor, and find peak value. */
    barea_inv = (jy_beam) ? (1.0 / barea) : 1.0;
    if (type == TFLOAT)
    {
        float *img = (float*)data;
        for (i = 0; i < num_pix; ++i)
        {
            img[i] *= barea_inv;
            val = img[i];
            if (val < noise_floor)
                img[i] = 0.0;
            else if (val > peak)
                peak = val;
        }
    }
    else if (type == TDOUBLE)
    {
        double *img = (double*)data;
        for (i = 0; i < num_pix; ++i)
        {
            img[i] *= barea_inv;
            val = img[i];
            if (val < noise_floor)
                img[i] = 0.0;
            else if (val > peak)
                peak = val;
        }
    }

    /* Convert reference values to radians. */
    crval[0] *= M_PI / 180.0;
    crval[1] *= M_PI / 180.0;

    /* Scale reference pixel values by downsample factor. */
    crpix[0] /= downsample_factor;
    crpix[1] /= downsample_factor;

    /* Compute sine of pixel deltas for inverse orthographic projection. */
    cdelt[0] = sin(downsample_factor * cdelt[0] * M_PI / 180.0);
    cdelt[1] = sin(downsample_factor * cdelt[1] * M_PI / 180.0);

    /* Down-sample and store the relevant image pixels. */
    width  = (naxes[0] + downsample_factor - 1) / downsample_factor;
    height = (naxes[1] + downsample_factor - 1) / downsample_factor;
    if (type == TFLOAT)
    {
        float *img = (float*)data, ra = 0.0, dec = 0.0, l = 0.0, m = 0.0;
        for (y = 0, j = 0; y < height; ++y)
        {
            for (x = 0; x < width; ++x)
            {
                /* Down-sample the image. */
                for (yy = 0, val_new = 0.0; yy < downsample_factor; ++yy)
                {
                    for (xx = 0; xx < downsample_factor; ++xx)
                    {
                        /* Calculate row & column indices and check ranges. */
                        ix = (xx + x * downsample_factor);
                        iy = (yy + y * downsample_factor);
                        if (ix >= naxes[0] || iy >= naxes[1])
                            continue;

                        /* Check pixel value before accumulation. */
                        val = img[(naxes[0] * iy) + ix];
                        if (val < peak * min_peak_fraction)
                            continue;
                        val_new += val;
                    }
                }
                if (val_new > 0.0)
                {
                    /* Convert pixel positions to RA and Dec values. */
                    l = cdelt[0] * (x + 1 - crpix[0]);
                    m = cdelt[1] * (y + 1 - crpix[1]);
                    oskar_convert_relative_directions_to_lon_lat_2d_f(1,
                            &l, &m, crval[0], crval[1], &ra, &dec);

                    /* Store pixel data in sky model. */
                    if (j % 100 == 0)
                        oskar_sky_resize(temp_sky, j + 100, &err);
                    oskar_sky_set_source(temp_sky, j, ra, dec,
                            val_new, 0.0, 0.0, 0.0, ref_freq, spectral_index,
                            0.0, 0.0, 0.0, 0.0, &err);
                    if (err) goto cleanup;
                    ++j;
                }
            }
        }
    }
    else if (type == TDOUBLE)
    {
        double *img = (double*)data, ra = 0.0, dec = 0.0, l = 0.0, m = 0.0;
        for (y = 0, j = 0; y < height; ++y)
        {
            for (x = 0; x < width; ++x)
            {
                /* Down-sample the image. */
                for (yy = 0, val_new = 0.0; yy < downsample_factor; ++yy)
                {
                    for (xx = 0; xx < downsample_factor; ++xx)
                    {
                        /* Calculate row & column indices and check ranges. */
                        ix = (xx + x * downsample_factor);
                        iy = (yy + y * downsample_factor);
                        if (ix >= naxes[0] || iy >= naxes[1])
                            continue;

                        /* Check pixel value before accumulation. */
                        val = img[(naxes[0] * iy) + ix];
                        if (val < peak * min_peak_fraction)
                            continue;
                        val_new += val;
                    }
                }
                if (val_new > 0.0)
                {
                    /* Convert pixel positions to RA and Dec values. */
                    l = cdelt[0] * (x + 1 - crpix[0]);
                    m = cdelt[1] * (y + 1 - crpix[1]);
                    oskar_convert_relative_directions_to_lon_lat_2d_d(1,
                            &l, &m, crval[0], crval[1], &ra, &dec);

                    /* Store pixel data in sky model. */
                    if (j % 100 == 0)
                        oskar_sky_resize(temp_sky, j + 100, &err);
                    oskar_sky_set_source(temp_sky, j, ra, dec,
                            val_new, 0.0, 0.0, 0.0, ref_freq, spectral_index,
                            0.0, 0.0, 0.0, 0.0, &err);
                    if (err) goto cleanup;
                    ++j;
                }
            }
        }
    }

    /* Set size to the number of elements loaded, and append to output. */
    oskar_sky_resize(temp_sky, j, &err);
    oskar_sky_append(sky, temp_sky, &err);
    oskar_log_message(ptr, 'M', 0, "Loaded %d pixels.", j);

    /* Close the FITS file and free memory. */
    cleanup:
    fits_close_file(fptr, &status);
    if (data) free(data);
    if (temp_sky) oskar_sky_free(temp_sky, &err);
    if (status) return OSKAR_ERR_FITS_IO;
    return err;
}

#ifdef __cplusplus
}
#endif
