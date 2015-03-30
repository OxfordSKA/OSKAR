/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include "fits/oskar_fits_healpix_to_sky_model.h"
#include <oskar_convert_healpix_ring_to_theta_phi.h>
#include <oskar_convert_galactic_to_fk5.h>
#include <oskar_log.h>
#include <oskar_sky.h>
#include <oskar_cmath.h>

#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fitsio.h>

#ifdef __cplusplus
extern "C" {
#endif

static const double boltzmann = 1.3806488e-23; /* Boltzmann constant in J/K. */

/*
 * TODO Possible enhancements:
 * 1) NEST to RING Conversion
 * 2) Remove implicit assumption that healpix data is in the first field of the
 *    binary table.
 * 2) Remove implicit assumption that only the first HDU of healpix data
 *    should be considered.
 *
 * Binary table mandatory keywords: (http://goo.gl/YT8Wbm)
 * 1. XTENSION ( == BINTABLE )
 * 2. BITPIX ( == 0 )
 * 3. NAXIS ( == 2 )
 * 4. NAXIS1
 * 5. NAXIS2
 * 6. PCOUNT
 * 7. GCOUNT
 * 8. TFIELDS
 * 9..N. TFORMn n=1..k where k = value of TFIELDS
 * last. END
 *
 * Expected HEALPIX keywords:
 * 1. PIXTYPE ( == HEALPIX )
 * 2. ORDERING ( == NESTED or RING )
 * 3. NSIDE
 * 4. FIRSTPIX?
 * 5. LASTPIX?
 * 6. INDXSCHM?
 */
void oskar_fits_healpix_to_sky_model(oskar_Log* ptr, const char* filename,
        const oskar_SettingsSkyHealpixFits* settings, oskar_Sky* sky,
        int* status)
{
    void* data = 0;
    int col_index = 1; /* Read the first column. */
    int type_code = 0, ncols = 0, num_hdu = 0, hdu = 0, nside = 0;
    long repeat = 0, width = 0, nrows = 0, i = 0, npixels = 0;
    fitsfile* fptr = 0;
    double lat = 0.0, lon = 0.0, val = 0.0;
    char card1[FLEN_CARD], card2[FLEN_CARD];
    oskar_Sky* temp_sky;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Open the FITS file. */
    fits_open_file(&fptr, filename, READONLY, status);
    if (*status)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Get number of HDUs in file. */
    fits_get_num_hdus(fptr, &num_hdu, status);

    /* Loop over extensions until a HEALPix binary table is found. */
    for (hdu = 2; hdu <= num_hdu; ++hdu)
    {
        int hdutype = 0;
        fits_movabs_hdu(fptr, hdu, &hdutype, status);
        if (hdutype != BINARY_TBL)
            continue;

        /* Look for PIXTYPE keyword. */
        fits_read_key(fptr, TSTRING, "XTENSION", card1, NULL, status);
        fits_read_key(fptr, TSTRING, "PIXTYPE", card2, NULL, status);
        if (!strcmp(card1, "BINTABLE") && !strcmp(card2, "HEALPIX"))
            break;
    }

    /* Check if any HDUs describe HEALPix data. */
    if (hdu > num_hdu)
    {
        fits_close_file(fptr, status);
        *status = OSKAR_ERR_FITS_IO;
        oskar_log_error(ptr, "Could not locate any HEALPix extensions.");
        return;
    }

    /* Check HEALPix ordering scheme. */
    fits_read_key(fptr, TSTRING, "ORDERING", card1, NULL, status);
    if (strcmp(card1, "RING"))
    {
        fits_close_file(fptr, status);
        *status = OSKAR_ERR_FITS_IO;
        oskar_log_error(ptr, "HEALPix data is not in RING format.");
        return;
    }

    /* Get number of rows and columns in binary table. */
    /* Number of rows = value of NAXIS2 keyword */
    /* Number of columns = value of TFIELDS keyword */
    fits_get_num_rows(fptr, &nrows, status);
    fits_get_num_cols(fptr, &ncols, status);

    /* Read type of the first column. */
    fits_get_coltype(fptr, col_index, &type_code, &repeat, &width, status);

    /* Number of pixels = number of rows x number of pixels in each row */
    npixels = nrows * repeat;

    /* Allocate memory for pixel data. */
    if (type_code == TFLOAT)
    {
        data = malloc(npixels * sizeof(float));
    }
    else if (type_code == TDOUBLE)
    {
        data = malloc(npixels * sizeof(double));
    }
    else
    {
        fits_close_file(fptr, status);
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        oskar_log_error(ptr, "Unsupported FITS data type.");
        return;
    }

    /* Get HEALPix NSIDE parameter. */
    fits_read_key(fptr, TINT, "NSIDE", &nside, NULL, status);
    if (npixels != 12 * nside * nside)
    {
        free(data);
        fits_close_file(fptr, status);
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        oskar_log_error(ptr, "HEALPix NSIDE parameter does not match number "
                "of rows in table.");
        return;
    }

    /* Read the FITS binary table into memory, and close the file. */
    fits_read_col(fptr, type_code, col_index, 1, 1, npixels, 0, data, 0, status);
    fits_close_file(fptr, status);

    /* Initialise the temporary sky model to hold all pixels in the table. */
    temp_sky = oskar_sky_create(oskar_sky_precision(sky),
            OSKAR_CPU, (int)npixels, status);

    /* Write contents of memory to temporary sky model. */
    for (i = 0; i < npixels; ++i)
    {
        val = (type_code == TFLOAT) ? ((float*)data)[i] : ((double*)data)[i];

        /* Convert map value into Stokes I flux in Jy, if required. */
        if (settings->map_units == OSKAR_MAP_UNITS_K_PER_SR ||
                settings->map_units == OSKAR_MAP_UNITS_MK_PER_SR)
        {
            /* FIXME Error - needs to be fixed. */
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
            break;

            /* Convert temperature per steradian to temperature per pixel. */
            /* Divide by number of pixels per steradian. */
            val /= (npixels / (4.0 * M_PI));
            if (settings->map_units == OSKAR_MAP_UNITS_MK_PER_SR)
            {
                val /= 1000.0; /* Convert milli-Kelvin to Kelvin. */
            }

            /* Convert temperature per pixel to Jansky per pixel. */
            /* Multiply by 2.0 * k_B * 10^26. */
            /* FIXME Assume that any wavelength dependence is already
             * in the input data! - Not true: need wavelength here too. */
            val *= (2.0 * boltzmann * 1e26);
        }

        /* Convert HEALPix index into spherical coordinates. */
        oskar_convert_healpix_ring_to_theta_phi_d(nside, i, &lat, &lon);
        lat = M_PI / 2.0 - lat;

        /* Convert spherical coordinates to RA, Dec values if required. */
        if (settings->coord_sys == OSKAR_SPHERICAL_TYPE_GALACTIC)
            oskar_convert_galactic_to_fk5_d(1, &lon, &lat, &lon, &lat);

        /* Set source data into sky model. */
        /* (Filtering and other overrides can be done later.) */
        oskar_sky_set_source(temp_sky, i, lon, lat, val,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, status);
    }

    /* Append temporary sky model to input data. */
    if (!(*status))
    {
        oskar_sky_append(sky, temp_sky, status);
        oskar_log_message(ptr, 'M', 0, "Loaded %d pixels.", (int)npixels);
    }

    /* Free memory. */
    if (data) free(data);
    oskar_sky_free(temp_sky, status);
}

#ifdef __cplusplus
}
#endif
