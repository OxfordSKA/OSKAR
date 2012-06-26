/*
 * Copyright (c) 2012, The University of Oxford
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

#include "fits/oskar_fits_to_sky_model.h"
#include "fits/oskar_fits_check_status.h"
#include "math/oskar_sph_from_lm.h"
#include "sky/oskar_sky_model_append.h"
#include "sky/oskar_sky_model_free.h"
#include "sky/oskar_sky_model_init.h"
#include "sky/oskar_sky_model_resize.h"
#include "sky/oskar_sky_model_set_source.h"
#include "sky/oskar_sky_model_write.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_log_error.h"
#include "utility/oskar_log_message.h"
#include "utility/oskar_log_warning.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fitsio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_AXES 10
#define FACTOR (2.0*sqrt(2.0*log(2.0)))

int oskar_fits_to_sky_model(oskar_Log* ptr, const char* filename,
        oskar_SkyModel* sky, double min_peak_fraction, double noise_floor)
{
    char value[FLEN_VALUE], card[FLEN_CARD];
    int i = 0, n = 0, num_pixels = 0, err = 0, status = 0, status2 = 0;
    int naxis = 0, datatype = 0, imagetype = 0, anynul = 0;
    int bytes_per_element = 0;
    int jy_beam = 0, jy_pixel = 0;
    fitsfile* fptr = NULL;
    double peakval = 0.0, nulval = 0.0;
    double crval1 = 0.0, crval2 = 0.0, crpix1 = 0.0, crpix2 = 0.0;
    double cdelt1 = 0.0, cdelt2 = 0.0, bmaj = 0.0, bmin = 0.0, beam_area = 0.0;
    long naxes[MAX_AXES];
    void* data = 0;
    oskar_SkyModel temp_sky;

    /* Initialise the temporary sky model. */
    oskar_sky_model_init(&temp_sky, sky->RA.type, OSKAR_LOCATION_CPU, 0);

    /* Open the FITS file. */
    fits_open_file(&fptr, filename, READONLY, &status);
    oskar_fits_check_status(ptr, status, "Opening file");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Get the image parameters. */
    fits_get_img_param(fptr, MAX_AXES, &imagetype, &naxis, naxes, &status);
    oskar_fits_check_status(ptr, status, "Reading image parameters");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Set the image data type. */
    if (imagetype == FLOAT_IMG)
    {
        datatype = TFLOAT;
        bytes_per_element = sizeof(float);
    }
    else if (imagetype == DOUBLE_IMG)
    {
        datatype = TDOUBLE;
        bytes_per_element = sizeof(double);
    }
    else
    {
        oskar_log_error(ptr, "Unknown FITS data type.");
        return OSKAR_ERR_FITS_IO;
    }

    /* Check that FITS image contains only two dimensions. */
    if (naxis > 2)
    {
        oskar_log_warning(ptr, "FITS image contains more than two dimensions. "
                "(Reading only the first plane.)");
    }
    else if (naxis < 2)
    {
        oskar_log_error(ptr, "This is not a recognised FITS image.");
        return OSKAR_ERR_FITS_IO;
    }

    /* Read and check CTYPE1. */
    fits_read_key(fptr, TSTRING, "CTYPE1", value, NULL, &status);
    oskar_fits_check_status(ptr, status, "Reading CTYPE1");
    if (status) return OSKAR_ERR_FITS_IO;
    if (strcmp(value, "RA---SIN") != 0)
    {
        oskar_log_error(ptr, "Unknown FITS axis 1 type.");
        return OSKAR_ERR_FITS_IO;
    }

    /* Read and check CTYPE2. */
    fits_read_key(fptr, TSTRING, "CTYPE2", value, NULL, &status);
    oskar_fits_check_status(ptr, status, "Reading CTYPE2");
    if (status) return OSKAR_ERR_FITS_IO;
    if (strcmp(value, "DEC--SIN") != 0)
    {
        oskar_log_error(ptr, "Unknown FITS axis 2 type.");
        return OSKAR_ERR_FITS_IO;
    }

    /* Read CDELT values. */
    fits_read_key(fptr, TDOUBLE, "CDELT1", &cdelt1, NULL, &status);
    oskar_fits_check_status(ptr, status, "Reading CDELT1");
    if (status) return OSKAR_ERR_FITS_IO;
    fits_read_key(fptr, TDOUBLE, "CDELT2", &cdelt2, NULL, &status);
    oskar_fits_check_status(ptr, status, "Reading CDELT2");
    if (status) return OSKAR_ERR_FITS_IO;
    if (fabs(cdelt1) - fabs(cdelt2) > 1e-8)
    {
        oskar_log_error(ptr, "Map pixels are not square.");
        return OSKAR_ERR_FITS_IO;
    }

    /* Read CRPIX values. */
    fits_read_key(fptr, TDOUBLE, "CRPIX1", &crpix1, NULL, &status);
    oskar_fits_check_status(ptr, status, "Reading CRPIX1");
    if (status) return OSKAR_ERR_FITS_IO;
    fits_read_key(fptr, TDOUBLE, "CRPIX2", &crpix2, NULL, &status);
    oskar_fits_check_status(ptr, status, "Reading CRPIX2");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Read CRVAL values. */
    fits_read_key(fptr, TDOUBLE, "CRVAL1", &crval1, NULL, &status);
    oskar_fits_check_status(ptr, status, "Reading CRVAL1");
    if (status) return OSKAR_ERR_FITS_IO;
    fits_read_key(fptr, TDOUBLE, "CRVAL2", &crval2, NULL, &status);
    oskar_fits_check_status(ptr, status, "Reading CRVAL2");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Read map units. */
    fits_read_key(fptr, TSTRING, "BUNIT", value, NULL, &status);
    oskar_fits_check_status(ptr, status, "Reading BUNIT");
    if (status)
    {
        oskar_log_error(ptr, "Could not determine map units.");
        return OSKAR_ERR_FITS_IO;
    }
    if (strcmp(value, "JY/BEAM") == 0)
    {
        jy_beam = 1;
    }
    else
    {
        if (strcmp(value, "JY/PIXEL") == 0)
            jy_pixel = 1;
        else
        {
            oskar_log_error(ptr, "Unknown units: need JY/BEAM or JY/PIXEL");
            return OSKAR_ERR_FITS_IO;
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
        oskar_log_message(ptr, 0, "Searching %d headers for beam size...",
                cards);
        for (i = 0; i < cards; ++i)
        {
            fits_read_record(fptr, i, card, &status);
            if (!strncmp(card, "HISTORY AIPS   CLEAN BMAJ", 25))
            {
                printf("%s\n", card);
                sscanf(card + 26, "%lf", &bmaj);
                sscanf(card + 44, "%lf", &bmin);
            }
        }
    }

    /* Check if we have beam size information. */
    if (bmaj > 0.0 && bmin > 0.0)
    {
        bmaj *= 3600.0;
        bmin *= 3600.0;
        oskar_log_message(ptr, 0, "Found beam size to be "
                "%.3f x %.3f arcsec.", bmaj, bmin);

        /* Calculate the beam area in pixels (normalisation factor). */
        beam_area = 2.0 * M_PI * (bmaj * bmin)
                    / (FACTOR * FACTOR * cdelt1 * cdelt1 * 3600.0 * 3600.0);
        oskar_log_message(ptr, 0, "Beam area is %.3f pixels.", beam_area);
    }
    else if (jy_beam)
    {
        oskar_log_error(ptr, "Unknown beam size, and map units are JY/BEAM.");
        return OSKAR_ERR_FITS_IO;
    }

    /* Allocate memory for image data. */
    num_pixels = naxes[0] * naxes[1];
    data = malloc(bytes_per_element * num_pixels);
    if (data == NULL)
        return OSKAR_ERR_MEMORY_ALLOC_FAILURE;

    /* Read image data. */
    fits_read_img(fptr, datatype, 1, num_pixels, &nulval, data, &anynul,
            &status);
    oskar_fits_check_status(ptr, status, "Reading image data");
    if (status) goto cleanup;

    /* Find peak value in image. */
    if (datatype == TFLOAT)
    {
        float* img = (float*)data;
        for (i = 0; i < num_pixels; ++i)
        {
            if (img[i] > peakval)
                peakval = img[i];
        }
    }
    else if (datatype == TDOUBLE)
    {
        double* img = (double*)data;
        for (i = 0; i < num_pixels; ++i)
        {
            if (img[i] > peakval)
                peakval = img[i];
        }
    }

    /* Populate the sky model. */
    if (datatype == TFLOAT)
    {
        float *img, ra = 0.0, dec = 0.0, val = 0.0, val_scaled = 0.0;
        float l = 0.0, m = 0.0, f1, f2;
        int x = 0, y = 0;
        img = (float*)data;
        f1 = sin(cdelt1 * M_PI / 180.0);
        f2 = sin(cdelt2 * M_PI / 180.0);
        for (i = 0, x = 0; x < naxes[0]; ++x)
        {
            for (y = 0; y < naxes[1]; ++y, ++i)
            {
                val = img[i];
                val_scaled = (jy_beam) ? val / beam_area : val;
                if (val > peakval * min_peak_fraction &&
                        val_scaled > noise_floor)
                {
                    /* Convert pixel positions to RA and Dec values. */
                    l = f1 * (x - crpix1);
                    m = f2 * (y - crpix2);
                    oskar_sph_from_lm_f(1, crval1 * M_PI / 180,
                            crval2 * M_PI / 180, &l, &m, &ra, &dec);

                    /* Store pixel data in sky model. */
                    /* Ensure enough space in arrays. */
                    if (n % 100 == 0)
                    {
                        err = oskar_sky_model_resize(&temp_sky, n + 100);
                        if (err) goto cleanup;
                    }
                    oskar_sky_model_set_source(&temp_sky, n, ra, dec,
                            val_scaled, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                    ++n;
                }
            }
        }
    }
    else if (datatype == TDOUBLE)
    {
    }

    /* Record the number of elements loaded. */
    temp_sky.num_sources = n;
    oskar_sky_model_append(sky, &temp_sky);

    /* Report the number of sources loaded. */
    oskar_log_message(ptr, 0, "Loaded %d sources from %s", n, filename);

    /* TEST write out sky model. */
    oskar_sky_model_write("test_fits_load.osm", sky);

    /* Close the FITS file and free memory. */
    cleanup:
    fits_close_file(fptr, &status);
    free(data);
    oskar_sky_model_free(&temp_sky);
    if (status) return OSKAR_ERR_FITS_IO;
    return err;
}

#ifdef __cplusplus
}
#endif
