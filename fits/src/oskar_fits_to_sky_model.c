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
#include "utility/oskar_Mem.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fitsio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_AXES 10

int oskar_fits_to_sky_model(oskar_Log* log, const char* filename,
        oskar_SkyModel* sky)
{
    char key[FLEN_KEYWORD], value[FLEN_VALUE];
    int i, num_elements = 1, status = 0, naxis = 0;
    int datatype = TFLOAT, imagetype = FLOAT_IMG, anynul = 0;
    int bytes_per_element = 0;
    fitsfile* fptr = NULL;
    double nulval = 0;
    long naxes[MAX_AXES];
    void* data = 0;

    /* Open the FITS file. */
    fits_open_file(&fptr, filename, READONLY, &status);
    oskar_fits_check_status(log, status, "Opening file");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Get the image parameters. */
    fits_get_img_param(fptr, MAX_AXES, &imagetype, &naxis, naxes, &status);
    oskar_fits_check_status(log, status, "Reading image parameters");
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
        oskar_log_error(log, "Unknown FITS data type.");
        return OSKAR_ERR_FITS_IO;
    }

    /* Check that FITS image contains only two dimensions. */
    if (naxis > 2)
    {
        oskar_log_warning(log, "FITS image contains more than two dimensions. "
                "(Reading only the first plane.)");
    }
    else if (naxis < 2)
    {
        oskar_log_error(log, "This is not a recognised FITS image.");
        return OSKAR_ERR_FITS_IO;
    }

    /* Read axis 1 type. */
    fits_read_key(fptr, TSTRING, "CTYPE1", value, NULL, &status);
    oskar_fits_check_status(log, status, "Reading axis 1 type");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Check axis 1 type. */
    if (strcmp(value, "RA---SIN") != 0)
    {
        oskar_log_error(log, "Unknown FITS axis 1 type.");
        return OSKAR_ERR_FITS_IO;
    }

    /* Read axis 2 type. */
    fits_read_key(fptr, TSTRING, "CTYPE2", value, NULL, &status);
    oskar_fits_check_status(log, status, "Reading axis 2 type");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Check axis 2 type. */
    if (strcmp(value, "DEC--SIN") != 0)
    {
        oskar_log_error(log, "Unknown FITS axis 2 type.");
        return OSKAR_ERR_FITS_IO;
    }

    /* Allocate memory for image data. */
    data = malloc(bytes_per_element * naxes[0] * naxes[1]);

    /* Read image data. */
    fits_read_img(fptr, datatype, 1, num_elements, &nulval, data, &anynul,
            &status);
    oskar_fits_check_status(log, status, "Reading image data");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Close the FITS file. */
    fits_close_file(fptr, &status);
    oskar_fits_check_status(log, status, "Closing file");
    if (status) return OSKAR_ERR_FITS_IO;

    cleanup:
    free(data);

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
