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

#include "fits/oskar_fits_image_write.h"
#include "fits/oskar_fits_write.h"
#include "fits/oskar_fits_write_axis_header.h"
#include "fits/oskar_fits_check_status.h"
#include "utility/oskar_Mem.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fitsio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_DIM 10

void oskar_fits_image_write(const oskar_Image* image, const char* filename)
{
    char key[FLEN_KEYWORD], value[FLEN_VALUE];
    int i, num_dimensions, status = 0, decimals = 10, type;
    long naxes[MAX_DIM];
    double crval[MAX_DIM], crpix[MAX_DIM], cdelt[MAX_DIM], crota[MAX_DIM];
    fitsfile* fptr = NULL;
    const char *label[MAX_DIM], *ctype[MAX_DIM];

    /* Get the data type. */
    type = image->data.type;

    /* Get the number of dimensions. */
    num_dimensions = image->dimension_order.num_elements;
    if (num_dimensions > 10)
        return;

    /* Loop over axes. */
    for (i = 0; i < num_dimensions; ++i)
    {
        int dim;
        dim = ((int*) image->dimension_order.data)[i];
        if (dim == OSKAR_IMAGE_DIM_RA)
        {
            label[i] = "Right Ascension";
            ctype[i] = "RA---SIN";
            naxes[i] = image->width;
            crval[i] = image->centre_ra_deg;
            cdelt[i] = -(image->fov_ra_deg / image->width); /* Convention */
            crpix[i] = (image->width + 1) / 2.0;
            crota[i] = 0.0;
        }
        else if (dim == OSKAR_IMAGE_DIM_DEC)
        {
            label[i] = "Declination";
            ctype[i] = "DEC--SIN";
            naxes[i] = image->height;
            crval[i] = image->centre_dec_deg;
            cdelt[i] = image->fov_dec_deg / image->height;
            crpix[i] = (image->height + 1) / 2.0;
            crota[i] = 0.0;
        }
        else if (dim == OSKAR_IMAGE_DIM_CHANNEL)
        {
            label[i] = "Frequency";
            ctype[i] = "FREQ";
            naxes[i] = image->num_channels;
            crval[i] = image->freq_start_hz;
            cdelt[i] = image->freq_inc_hz;
            crpix[i] = 1.0;
            crota[i] = 0.0;
        }
        else if (dim == OSKAR_IMAGE_DIM_POL)
        {
            label[i] = "Polarisation";
            ctype[i] = "STOKES";
            naxes[i] = image->num_pols;
            crval[i] = 1.0;
            cdelt[i] = 1.0;
            crpix[i] = 1.0;
            crota[i] = 0.0;
        }
        else if (dim == OSKAR_IMAGE_DIM_TIME)
        {
            label[i] = "Time";
            ctype[i] = "UTC";
            naxes[i] = image->num_times;
            crval[i] = 1.0;
            cdelt[i] = 1.0;
            crpix[i] = 1.0;
            crota[i] = 0.0;
        }
    }

    /* Write multi-dimensional image data. */
    oskar_fits_write(filename, type, num_dimensions, naxes, image->data.data, ctype,
            label, crval, cdelt, crpix, crota);

    /* Open file for read/write access. */
    fits_open_file(&fptr, filename, READWRITE, &status);

    /* Write brightness unit keyword. */
    strcpy(key, "BUNIT");
    strcpy(value, "JY/BEAM");
    fits_write_key_str(fptr, key, value, "Units of flux", &status);
    oskar_fits_check_status(status, "Writing key: BUNIT");

    /* Write pointing keywords. */
    fits_write_key_dbl(fptr, "OBSRA", image->centre_ra_deg, decimals,
            "Pointing RA", &status);
    oskar_fits_check_status(status, "Writing key: OBSRA");
    fits_write_key_dbl(fptr, "OBSDEC", image->centre_dec_deg, decimals,
            "Pointing DEC", &status);
    oskar_fits_check_status(status, "Writing key: OBSDEC");

    /* Close the FITS file. */
    fits_close_file(fptr, &status);
    oskar_fits_check_status(status, "Closing file");
}

#ifdef __cplusplus
}
#endif
