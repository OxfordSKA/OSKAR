/*
 * Copyright (c) 2016, The University of Oxford
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

#include "mem/oskar_mem.h"
#include <fitsio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Mem* oskar_mem_read_healpix_fits(const char* filename,
        int healpix_hdu_index, int* nside, char* ordering, char* coordsys,
        char** brightness_units, int* status)
{
    int col_index = 1; /* Read the first column. */
    int i_healpix = 0, i_hdu = 0, len, num_cols = 0, num_hdu = 0;
    int type_fits = 0, type_oskar = 0, status1 = 0;
    long repeat = 0, width = 0, num_rows = 0, num_pixels = 0;
    char card1[FLEN_CARD], card2[FLEN_CARD];
    fitsfile* fptr = 0;
    oskar_Mem* data = 0;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Open the FITS file. */
    fits_open_file(&fptr, filename, READONLY, status);
    if (*status || !fptr)
    {
        *status = OSKAR_ERR_FILE_IO;
        return 0;
    }

    /* Get number of HDUs in file. */
    fits_get_num_hdus(fptr, &num_hdu, status);

    /* Loop over extensions until the right HEALPix binary table is found. */
    for (i_hdu = 2; i_hdu <= num_hdu; ++i_hdu)
    {
        int hdutype = 0;
        fits_movabs_hdu(fptr, i_hdu, &hdutype, status);
        if (hdutype != BINARY_TBL)
            continue;

        /* Look for PIXTYPE keyword. */
        fits_read_key(fptr, TSTRING, "XTENSION", card1, NULL, status);
        fits_read_key(fptr, TSTRING, "PIXTYPE", card2, NULL, status);
        if (!strcmp(card1, "BINTABLE") && !strcmp(card2, "HEALPIX"))
        {
            if (i_healpix == healpix_hdu_index)
                break;
            else
                i_healpix++;
        }
    }

    /* Check if we exhausted the HDU list without finding anything. */
    if (i_hdu > num_hdu)
    {
        fits_close_file(fptr, status);
        *status = OSKAR_ERR_FILE_IO;
        return 0;
    }

    /* Get number of rows and columns in binary table. */
    fits_get_num_rows(fptr, &num_rows, status);
    fits_get_num_cols(fptr, &num_cols, status);

    /* Read type of the first column. */
    fits_get_coltype(fptr, col_index, &type_fits, &repeat, &width, status);
    type_oskar = (type_fits == TDOUBLE) ? OSKAR_DOUBLE : OSKAR_SINGLE;

    /* Number of pixels = number of rows x number of pixels in each row */
    num_pixels = num_rows * repeat;

    /* Get HEALPix NSIDE parameter. */
    fits_read_key(fptr, TINT, "NSIDE", nside, NULL, status);
    if (num_pixels != 12 * (*nside) * (*nside))
    {
        fits_close_file(fptr, status);
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return 0;
    }

    /* Get HEALPix ordering scheme. */
    card1[0] = 0;
    fits_read_key(fptr, TSTRING, "ORDERING", card1, NULL, status);
    *ordering = card1[0];

    /* Get coordinate system. */
    card1[0] = 0;
    fits_read_key(fptr, TSTRING, "COORDSYS", card1, NULL, status);
    *coordsys = card1[0];

    /* Get brightness units if present. */
    card1[0] = 0;
    status1 = 0;
    fits_read_key(fptr, TSTRING, "TUNIT1", card1, NULL, &status1);
    len = (int) strlen(card1);
    if (!status1 && len > 0)
    {
        *brightness_units = (char*) realloc (*brightness_units, len + 1);
        strcpy(*brightness_units, card1);
    }

    /* Read the FITS binary table into memory, and close the file. */
    data = oskar_mem_create(type_oskar, OSKAR_CPU, num_pixels, status);
    fits_read_col(fptr, type_fits, col_index, 1, 1, num_pixels, 0,
            oskar_mem_void(data), 0, status);
    fits_close_file(fptr, status);

    return data;
}

#ifdef __cplusplus
}
#endif
