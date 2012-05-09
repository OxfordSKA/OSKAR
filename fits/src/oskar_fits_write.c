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

int oskar_fits_write(const char* filename, int type, int naxis,
        long* naxes, void* data, const char** ctype, const char** ctype_desc,
        const double* crval, const double* cdelt, const double* crpix,
        const double* crota)
{
    char key[FLEN_KEYWORD], value[FLEN_VALUE];
    int i, num_elements = 1, status = 0;
    int datatype = TFLOAT, imagetype = FLOAT_IMG;
    fitsfile* fptr = NULL;

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

    /* Create a new empty output FITS file. */
    fits_create_file(&fptr, filename, &status);
    oskar_fits_check_status(status, "Creating file");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Create the new image. */
    fits_create_img(fptr, imagetype, naxis, naxes, &status);
    oskar_fits_check_status(status, "Creating image");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Write date stamp. */
    fits_write_date(fptr, &status);
    oskar_fits_check_status(status, "Writing date");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Write telescope keyword. */
    strcpy(key, "TELESCOP");
    strcpy(value, "OSKAR-2 SIMULATOR");
    fits_write_key_str(fptr,  key, value, NULL, &status);
    oskar_fits_check_status(status, "Writing key: TELESCOP");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Axis description headers. */
    for (i = 0; i < naxis; ++i)
    {
        oskar_fits_write_axis_header(fptr, i + 1, ctype[i], ctype_desc[i],
                crval[i], cdelt[i], crpix[i], crota[i]);
    }

    /* Write a history line. */
    fits_write_history(fptr,
            "This file was created using the OSKAR-2 simulator.", &status);
    oskar_fits_check_status(status, "Writing history");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Write image data. */
    for (i = 0; i < naxis; ++i)
    {
        num_elements *= naxes[i];
    }
    fits_write_img(fptr, datatype, 1, num_elements, data, &status);
    oskar_fits_check_status(status, "Writing image data");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Close the FITS file. */
    fits_close_file(fptr, &status);
    oskar_fits_check_status(status, "Closing file");
    if (status) return OSKAR_ERR_FITS_IO;

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
