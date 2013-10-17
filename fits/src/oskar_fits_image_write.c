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

#include <fits/oskar_fits_image_write.h>
#include <fits/oskar_fits_write_axis_header.h>
#include <oskar_file_exists.h>
#include <oskar_getline.h>
#include <oskar_log.h>
#include <oskar_mem.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fitsio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_DIM 9

void oskar_fits_image_write(oskar_Image* image, oskar_Log* log,
        const char* filename, int* status)
{
    char value[FLEN_VALUE];
    int i, num_dimensions, num_elements = 1, decimals = 10;
    int type, datatype, imagetype;
    long naxes[MAX_DIM], naxes_dummy[MAX_DIM];
    double crval[MAX_DIM], crpix[MAX_DIM], cdelt[MAX_DIM], crota[MAX_DIM];
    fitsfile* fptr = NULL;
    const char *label[MAX_DIM], *ctype[MAX_DIM];

    /* Check all inputs. */
    if (!image || !filename || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Set dummy axis sizes to 1. */
    for (i = 0; i < MAX_DIM; ++i)
    {
        naxes_dummy[i] = 1;
    }

    /* Get the data type. */
    type = oskar_mem_precision(&image->data);

    /* Get the number of dimensions. */
    num_dimensions = sizeof(image->dimension_order) / sizeof(int);
    if (num_dimensions > MAX_DIM)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Loop over axes. */
    for (i = 0; i < num_dimensions; ++i)
    {
        int dim;
        dim = image->dimension_order[i];
        if (dim == OSKAR_IMAGE_DIM_RA)
        {
            double max, inc, delta;

            /* Compute pixel delta. */
            max = sin(image->fov_ra_deg * M_PI / 360.0); /* Divide by 2. */
            inc = max / (0.5 * image->width);
            delta = -asin(inc) * 180.0 / M_PI; /* Negative convention. */

            /* Set axis properties. */
            label[i] = "Right Ascension";
            ctype[i] = "RA---SIN";
            naxes[i] = image->width;
            crval[i] = image->centre_ra_deg;
            cdelt[i] = delta;
            crpix[i] = (image->width + 1) / 2.0;
            crota[i] = 0.0;
        }
        else if (dim == OSKAR_IMAGE_DIM_DEC)
        {
            double max, inc, delta;

            /* Compute pixel delta. */
            max = sin(image->fov_dec_deg * M_PI / 360.0); /* Divide by 2. */
            inc = max / (0.5 * image->height);
            delta = asin(inc) * 180.0 / M_PI;

            /* Set axis properties. */
            label[i] = "Declination";
            ctype[i] = "DEC--SIN";
            naxes[i] = image->height;
            crval[i] = image->centre_dec_deg;
            cdelt[i] = delta;
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
            crval[i] = 0.0; /* Zero relative to MJD-OBS. */
            cdelt[i] = image->time_inc_sec;
            crpix[i] = 1.0;
            crota[i] = 0.0;
        }
    }

    /* If the file exists, remove it. */
    if (oskar_file_exists(filename))
        remove(filename);

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

    /* Create a new (empty) FITS file, and write the image headers
     * using dummy dimension values. */
    fits_create_file(&fptr, filename, status);
    fits_create_img(fptr, imagetype, num_dimensions, naxes_dummy, status);

    /* Write date stamp. */
    fits_write_date(fptr, status);

    /* Write telescope keyword. */
    strcpy(value, "OSKAR " OSKAR_VERSION_STR);
    fits_write_key_str(fptr, "TELESCOP", value, NULL, status);

    /* Axis description headers. */
    for (i = 0; i < num_dimensions; ++i)
    {
        oskar_fits_write_axis_header(fptr, i + 1, ctype[i], label[i],
                crval[i], cdelt[i], crpix[i], crota[i], status);
    }

    /* Write a history line with the OSKAR version. */
    fits_write_history(fptr,
            "This file was created using OSKAR " OSKAR_VERSION_STR, status);

    /* Write brightness unit keyword. */
    if (image->image_type < 10)
    {
        strcpy(value, "JY/BEAM");
        fits_write_key_str(fptr, "BUNIT", value, "Units of flux", status);
    }

    /* Write time header keywords. */
    strcpy(value, "UTC");
    fits_write_key_str(fptr, "TIMESYS", value, NULL, status);
    strcpy(value, "s");
    fits_write_key_str(fptr, "TIMEUNIT", value, "Time axis units", status);
    fits_write_key_dbl(fptr, "MJD-OBS", image->time_start_mjd_utc, decimals,
            "Obs start time", status);

    /* Write pointing keywords. */
    fits_write_key_dbl(fptr, "OBSRA", image->centre_ra_deg, decimals,
            "Pointing RA", status);
    fits_write_key_dbl(fptr, "OBSDEC", image->centre_dec_deg, decimals,
            "Pointing DEC", status);

    /* Write log entries as FITS HISTORY keys. */
    if (log && oskar_log_file_handle(log) && !(*status))
    {
        char* buffer = NULL;
        size_t buf_size = 0;
        FILE* fhan = 0;
        fhan = oskar_log_file_handle(log);
        fseek(fhan, 0, SEEK_SET);
        while (oskar_getline(&buffer, &buf_size, fhan) != OSKAR_ERR_EOF)
        {
            fits_write_history(fptr, buffer, status);
        }
        if (buffer) free(buffer);
    }

    /* Write image data into primary array. */
    for (i = 0; i < num_dimensions; ++i)
    {
        num_elements *= naxes[i];
    }
    fits_write_img(fptr, datatype, 1, num_elements,
            oskar_mem_void(&image->data), status);

    /* Update header keywords with the correct axis lengths.
     * Needs to be done here because CFITSIO doesn't let us write only the
     * header with the correct axis lengths to start with: it wastefully
     * tries to write a huge block of empty memory too! */
    strcpy(value, "NAXIS ");
    for (i = 0; i < num_dimensions; ++i)
    {
        value[5] = 49 + i; /* Index to ASCII character for i + 1. */
        fits_update_key_lng(fptr, value, naxes[i], 0, status);
    }

    /* Close the FITS file. */
    fits_close_file(fptr, status);
    if (*status)
        *status = OSKAR_ERR_FITS_IO;
}

#ifdef __cplusplus
}
#endif
