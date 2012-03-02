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

void oskar_fits_write_image(const char* filename, int type, int width,
        int height, void* data, double ra, double dec, double pixel_scale,
        double frequency, double bandwidth)
{
    char key[FLEN_KEYWORD], value[FLEN_VALUE];
    int status = 0, decimals = 10;
    long naxes[4];
    double crval[4], crpix[4], cdelt[4], crota[4];
    fitsfile* fptr = NULL;
    const char **ctype, **ctype_comment;

    /* Axis types. */
    ctype = (const char**) calloc(4, sizeof(const char*));
    ctype[0] = "RA---SIN";
    ctype[1] = "DEC--SIN";
    ctype[2] = "FREQ";
    ctype[3] = "STOKES";

    /* Axis comments. */
    ctype_comment = (const char**) calloc(4, sizeof(const char*));
    ctype_comment[0] = "Right Ascension";
    ctype_comment[1] = "Declination";
    ctype_comment[2] = "Frequency";
    ctype_comment[3] = "Polarisation";

    /* Axis dimensions. */
    naxes[0] = width;
    naxes[1] = height;
    naxes[2] = 1;
    naxes[3] = 1;

    /* Reference values. */
    crval[0] = ra;
    crval[1] = dec;
    crval[2] = frequency;
    crval[3] = 1.0;

    /* Deltas. */
    cdelt[0] = -pixel_scale;
    cdelt[1] = pixel_scale;
    cdelt[2] = bandwidth;
    cdelt[3] = 1.0;

    /* Reference pixels. */
    crpix[0] = (width + 1) / 2.0;
    crpix[1] = (height + 1) / 2.0;
    crpix[2] = 1.0;
    crpix[3] = 1.0;

    /* Rotation. */
    crota[0] = 0.0;
    crota[1] = 0.0;
    crota[2] = 0.0;
    crota[3] = 0.0;

    /* Write multi-dimensional image data. */
    oskar_fits_write(filename, type, 4, naxes, data, ctype, ctype_comment,
            crval, cdelt, crpix, crota);

    /* Open file for read/write access. */
    fits_open_file(&fptr, filename, READWRITE, &status);

    /* Write brightness unit keyword. */
    strcpy(key, "BUNIT");
    strcpy(value, "JY/BEAM");
    fits_write_key_str(fptr, key, value, "Units of flux", &status);
    oskar_fits_check_status(status, "Writing key: BUNIT");

    /* Write pointing keywords. */
    fits_write_key_dbl(fptr, "OBSRA", ra, decimals, "Pointing RA", &status);
    oskar_fits_check_status(status, "Writing key: OBSRA");
    fits_write_key_dbl(fptr, "OBSDEC", dec, decimals, "Pointing DEC", &status);
    oskar_fits_check_status(status, "Writing key: OBSDEC");

    /* Close the FITS file. */
    fits_close_file(fptr, &status);
    oskar_fits_check_status(status, "Closing file");

    /* Free memory. */
    free(ctype);
    free(ctype_comment);
}

#ifdef __cplusplus
}
#endif
