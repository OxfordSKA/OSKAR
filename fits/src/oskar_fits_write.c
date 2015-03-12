/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include <fits/oskar_fits_write.h>
#include <fits/oskar_fits_write_axis_header.h>
#include <oskar_mem.h>
#include <oskar_version.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fitsio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_fits_write(const char* filename, int type, int naxis,
        long* naxes, void* data, const char** ctype, const char** ctype_desc,
        const double* crval, const double* cdelt, const double* crpix,
        const double* crota, int* status)
{
    char value[FLEN_VALUE];
    int i, num_elements = 1;
    int datatype = TFLOAT, imagetype = FLOAT_IMG;
    fitsfile* fptr = NULL;
    FILE* file;

    /* Check if safe to proceed. */
    if (*status) return;

    /* If the file exists, remove it. */
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

    /* Create a new empty output FITS file. */
    fits_create_file(&fptr, filename, status);

    /* Create the new image. */
    fits_create_img(fptr, imagetype, naxis, naxes, status);

    /* Write date stamp. */
    fits_write_date(fptr, status);

    /* Write telescope keyword. */
    strcpy(value, "OSKAR " OSKAR_VERSION_STR);
    fits_write_key_str(fptr, "TELESCOP", value, NULL, status);

    /* Axis description headers. */
    for (i = 0; i < naxis; ++i)
    {
        oskar_fits_write_axis_header(fptr, i + 1, ctype[i], ctype_desc[i],
                crval[i], cdelt[i], crpix[i], crota[i], status);
    }

    /* Write a history line with the OSKAR version. */
    fits_write_history(fptr,
            "This file was created using OSKAR " OSKAR_VERSION_STR, status);

    /* Write image data into primary array. */
    for (i = 0; i < naxis; ++i)
    {
        num_elements *= naxes[i];
    }
    fits_write_img(fptr, datatype, 1, num_elements, data, status);

    /* Close the FITS file. */
    fits_close_file(fptr, status);
    if (*status) *status = OSKAR_ERR_FITS_IO;
}

#ifdef __cplusplus
}
#endif
