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
#include "utility/oskar_log_error.h"
#include "utility/oskar_log_warning.h"

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
    char value[FLEN_VALUE];
    int num_elements = 1, status = 0, naxis = 0;
    int datatype = TFLOAT, imagetype = FLOAT_IMG, anynul = 0;
    int bytes_per_element = 0;
    fitsfile* fptr = NULL;
    double nulval = 0.0;
    double crval1 = 0.0, crval2 = 0.0, crpix1 = 0.0, crpix2 = 0.0;
    double cdelt1 = 0.0, cdelt2 = 0.0;
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

    /* Read and check CTYPE1. */
    fits_read_key(fptr, TSTRING, "CTYPE1", value, NULL, &status);
    oskar_fits_check_status(log, status, "Reading CTYPE1");
    if (status) return OSKAR_ERR_FITS_IO;
    if (strcmp(value, "RA---SIN") != 0)
    {
        oskar_log_error(log, "Unknown FITS axis 1 type.");
        return OSKAR_ERR_FITS_IO;
    }

    /* Read and check CTYPE2. */
    fits_read_key(fptr, TSTRING, "CTYPE2", value, NULL, &status);
    oskar_fits_check_status(log, status, "Reading CTYPE2");
    if (status) return OSKAR_ERR_FITS_IO;
    if (strcmp(value, "DEC--SIN") != 0)
    {
        oskar_log_error(log, "Unknown FITS axis 2 type.");
        return OSKAR_ERR_FITS_IO;
    }

    /* Read CDELT values. */
    fits_read_key(fptr, TDOUBLE, "CDELT1", &cdelt1, NULL, &status);
    oskar_fits_check_status(log, status, "Reading CDELT1");
    if (status) return OSKAR_ERR_FITS_IO;
    fits_read_key(fptr, TDOUBLE, "CDELT2", &cdelt2, NULL, &status);
    oskar_fits_check_status(log, status, "Reading CDELT2");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Read CRPIX values. */
    fits_read_key(fptr, TDOUBLE, "CRPIX1", &crpix1, NULL, &status);
    oskar_fits_check_status(log, status, "Reading CRPIX1");
    if (status) return OSKAR_ERR_FITS_IO;
    fits_read_key(fptr, TDOUBLE, "CRPIX2", &crpix2, NULL, &status);
    oskar_fits_check_status(log, status, "Reading CRPIX2");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Read CRVAL values. */
    fits_read_key(fptr, TDOUBLE, "CRVAL1", &crval1, NULL, &status);
    oskar_fits_check_status(log, status, "Reading CRVAL1");
    if (status) return OSKAR_ERR_FITS_IO;
    fits_read_key(fptr, TDOUBLE, "CRVAL2", &crval2, NULL, &status);
    oskar_fits_check_status(log, status, "Reading CRVAL2");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Allocate memory for image data. */
    num_elements = naxes[0] * naxes[1];
    data = malloc(bytes_per_element * num_elements);
    if (data == NULL)
        return OSKAR_ERR_MEMORY_ALLOC_FAILURE;

    /* Read image data. */
    fits_read_img(fptr, datatype, 1, num_elements, &nulval, data, &anynul,
            &status);
    oskar_fits_check_status(log, status, "Reading image data");
    if (status) goto cleanup;

    /* TODO Populate the sky model. */
    sky->num_sources = num_elements;

    /* Close the FITS file and free memory. */
    cleanup:
    fits_close_file(fptr, &status);
    oskar_fits_check_status(log, status, "Closing file");
    free(data);
    if (status) return OSKAR_ERR_FITS_IO;
    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
