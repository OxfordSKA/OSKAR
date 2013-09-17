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
#include <fits/oskar_fits_write.h>
#include <fits/oskar_fits_write_axis_header.h>
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

#define MAX_DIM 10

void oskar_fits_image_write(const oskar_Image* image, oskar_Log* log,
        const char* filename, int* status)
{
    char value[FLEN_VALUE];
    int i, num_dimensions, decimals = 10, type;
    long naxes[MAX_DIM];
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

    /* Get the data type. */
    type = oskar_mem_type(&image->data);

    /* Get the number of dimensions. */
    num_dimensions = sizeof(image->dimension_order) / sizeof(int);
    if (num_dimensions > 10)
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

    /* Write multi-dimensional image data. */
    oskar_fits_write(filename, type, num_dimensions, naxes,
            image->data.data, ctype, label, crval, cdelt, crpix, crota, status);
    if (*status) return;

    /* Open file for read/write access. */
    fits_open_file(&fptr, filename, READWRITE, status);

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
    if (*status)
    {
        *status = OSKAR_ERR_FITS_IO;
        return;
    }

    /* Write log entries as FITS HISTORY keys. */
    if (log && oskar_log_file_handle(log))
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

    /* Close the FITS file. */
    fits_close_file(fptr, status);
    if (*status)
        *status = OSKAR_ERR_FITS_IO;
}

#ifdef __cplusplus
}
#endif
