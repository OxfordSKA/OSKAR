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

#include "fits/oskar_fits_write_image.h"
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

void oskar_fits_write_image(const char* filename, int type, int width,
        int height, void* data, double ra, double dec, double d_ra,
        double d_dec, double frequency, double bandwidth)
{
    char key[FLEN_KEYWORD], value[FLEN_VALUE];
    int status = 0, decimals = 10;
    int datatype = TFLOAT, imagetype = FLOAT_IMG;
    long naxes[4];
    fitsfile* fptr;

    /* If the file exists, remove it. */
    FILE* file;
    if ((file = fopen(filename, "r")) != NULL)
    {
        fclose(file);
        remove(filename);
    }

    /* Set the type data. */
    if (type == OSKAR_SINGLE)
    {
        datatype = TFLOAT;
        imagetype = FLOAT_IMG;
    }
    else if (type == OSKAR_DOUBLE)
    {
        datatype = TDOUBLE;
        imagetype = DOUBLE_IMG;
    }

    /* Set the dimensions. */
    naxes[0] = width;
    naxes[1] = height;
    naxes[2] = 1;
    naxes[3] = 1;

    /* Create a new empty output FITS file. */
    fits_create_file(&fptr, filename, &status);
    oskar_fits_check_status(status, "Creating file");

    /* Create the new image. */
    fits_create_img(fptr, imagetype, 4, naxes, &status);
    oskar_fits_check_status(status, "Creating image");

    /* Write date stamp. */
    fits_write_date(fptr, &status);
    oskar_fits_check_status(status, "Writing date");

    /* Write telescope keyword. */
    strcpy(key, "TELESCOP");
    strcpy(value, "OSKAR SIM (0.0.0)");
    fits_write_key_str(fptr,  key, value, NULL, &status);
    oskar_fits_check_status(status, "Writing key: TELESCOP");

    /* Write brightness unit keyword. */
    strcpy(key, "BUNIT");
    strcpy(value, "JY/BEAM");
    fits_write_key_str(fptr, key, value, "Units of flux", &status);
    oskar_fits_check_status(status, "Writing key: BUNIT");

    /* Write coordinate keywords. */
    fits_write_key_dbl(fptr, "EQUINOX", 2000.0, decimals, "Epoch of RA DEC",
            &status);
    oskar_fits_check_status(status, "Writing key: EQUINOX");
    fits_write_key_dbl(fptr, "OBSRA", ra, decimals, "Pointing RA", &status);
    oskar_fits_check_status(status, "Writing key: OBSRA");
    fits_write_key_dbl(fptr, "OBSDEC", dec, decimals, "Pointing DEC", &status);
    oskar_fits_check_status(status, "Writing key: OBSDEC");

    /* Axis description headers. */
    oskar_fits_write_axis_header(fptr, 1, "RA---SIN", "Right Ascension",
            ra, d_ra, width / 2.0, 0.0);
    oskar_fits_write_axis_header(fptr, 2, "DEC--SIN", "Declination",
            dec, d_dec, height / 2.0, 0.0);
    oskar_fits_write_axis_header(fptr, 3, "FREQ", "Frequency",
            frequency, bandwidth, 1.0, 0.0);
    oskar_fits_write_axis_header(fptr, 4, "STOKES", "Stokes",
            1.0, 1.0, 1.0, 0.0);

    /* Write a history line. */
    fits_write_history(fptr,
            "This image was created using the OSKAR-2 simulator.", &status);
    oskar_fits_check_status(status, "Writing history");

    /* Write image data. */
    fits_write_img(fptr, datatype, 1, width * height, data, &status);
    oskar_fits_check_status(status, "Writing image data");

    /* Close the FITS file. */
    fits_close_file(fptr, &status);
    oskar_fits_check_status(status, "Closing file");
}

#ifdef __cplusplus
}
#endif
