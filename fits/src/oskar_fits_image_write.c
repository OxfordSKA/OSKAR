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

#include "fits/oskar_fits_image_write.h"
#include "fits/oskar_fits_write.h"
#include "fits/oskar_fits_write_axis_header.h"
#include "fits/oskar_fits_check_status.h"
#include "utility/oskar_getline.h"
#include "utility/oskar_log_message.h"
#include "utility/oskar_Mem.h"

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

int oskar_fits_image_write(const oskar_Image* image, oskar_Log* log,
        const char* filename)
{
    char value[FLEN_VALUE];
    int i, num_dimensions, status = 0, decimals = 10, type;
    long naxes[MAX_DIM];
    double crval[MAX_DIM], crpix[MAX_DIM], cdelt[MAX_DIM], crota[MAX_DIM];
    fitsfile* fptr = NULL;
    const char *label[MAX_DIM], *ctype[MAX_DIM];

    /* Get the data type. */
    type = image->data.type;

    /* Get the number of dimensions. */
    num_dimensions = sizeof(image->dimension_order) / sizeof(int);
    if (num_dimensions > 10)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    /* Write a log message. */
    oskar_log_message(log, 0, "Writing FITS image file: '%s'", filename);

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
    status = oskar_fits_write(log, filename, type, num_dimensions, naxes,
            image->data.data, ctype, label, crval, cdelt, crpix, crota);
    if (status) return OSKAR_ERR_FITS_IO;

    /* Open file for read/write access. */
    fits_open_file(&fptr, filename, READWRITE, &status);
    oskar_fits_check_status(log, status, "Opening FITS file.");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Write brightness unit keyword. */
    if (image->image_type < 10)
    {
        strcpy(value, "JY/BEAM");
        fits_write_key_str(fptr, "BUNIT", value, "Units of flux", &status);
        oskar_fits_check_status(log, status, "Writing key: BUNIT");
        if (status) return OSKAR_ERR_FITS_IO;
    }

    /* Write time header keywords. */
    strcpy(value, "UTC");
    fits_write_key_str(fptr, "TIMESYS", value, NULL, &status);
    oskar_fits_check_status(log, status, "Writing key: TIMESYS");
    if (status) return OSKAR_ERR_FITS_IO;
    strcpy(value, "s");
    fits_write_key_str(fptr, "TIMEUNIT", value, "Time axis units", &status);
    oskar_fits_check_status(log, status, "Writing key: TIMEUNIT");
    if (status) return OSKAR_ERR_FITS_IO;
    fits_write_key_dbl(fptr, "MJD-OBS", image->time_start_mjd_utc, decimals,
            "Obs start time", &status);
    oskar_fits_check_status(log, status, "Writing key: MJD-OBS");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Write pointing keywords. */
    fits_write_key_dbl(fptr, "OBSRA", image->centre_ra_deg, decimals,
            "Pointing RA", &status);
    oskar_fits_check_status(log, status, "Writing key: OBSRA");
    if (status) return OSKAR_ERR_FITS_IO;
    fits_write_key_dbl(fptr, "OBSDEC", image->centre_dec_deg, decimals,
            "Pointing DEC", &status);
    oskar_fits_check_status(log, status, "Writing key: OBSDEC");
    if (status) return OSKAR_ERR_FITS_IO;

    /* Write log entries as FITS HISTORY keys. */
    if (log)
    {
        if (log->file)
        {
            char* log_line_buffer = NULL;
            size_t buffer_size = 0;
            fseek(log->file, 0, SEEK_SET);
            while (oskar_getline(&log_line_buffer,
                    &buffer_size, log->file) != OSKAR_ERR_EOF)
            {
                fits_write_history(fptr, log_line_buffer, &status);
            }
            if (log_line_buffer) free(log_line_buffer);
        }
    }

    /* Close the FITS file. */
    fits_close_file(fptr, &status);
    oskar_fits_check_status(log, status, "Closing file");
    if (status) return OSKAR_ERR_FITS_IO;

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
