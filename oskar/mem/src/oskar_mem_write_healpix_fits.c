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
#include "utility/oskar_file_exists.h"
#include <string.h>
#include <fitsio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_write_healpix_fits(oskar_Mem* data, const char* filename,
        int overwrite, int nside, char ordering, char coordsys, int* status)
{
    fitsfile* fptr = 0;
    char* ttype[] = { "SIGNAL" };
    char* tform[] = { "1?" };
    char coord[]  = { 0, 0 };
    char order[]  = { 0, 0, 0, 0, 0, 0, 0, 0 };

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check dimensions. */
    if ((int)oskar_mem_length(data) < 12 * nside * nside)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Set the binary table column type code. */
    tform[0][1] = oskar_mem_is_double(data) ? 'D' : 'E';

    /* Set the ordering scheme and the coordinate system. */
    coord[0] = coordsys;
    if (ordering == 'R' || ordering == 'r')
        strcpy(order, "RING");
    else if (ordering == 'N' || ordering == 'n')
        strcpy(order, "NESTED");

    /* Open or create the file as needed. */
    if (oskar_file_exists(filename))
    {
        if (overwrite)
        {
            remove(filename);
            fits_create_file(&fptr, filename, status);
        }
        else
        {
            fits_open_file(&fptr, filename, READWRITE, status);
        }
    }
    else
    {
        fits_create_file(&fptr, filename, status);
    }

    /* Check for errors. */
    if (*status || !fptr)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Write the data into a new binary table. */
    fits_create_tbl(fptr, BINARY_TBL, 12L * nside * nside, 1,
            ttype, tform, NULL, "BINTABLE", status);
    fits_write_key(fptr, TSTRING, "PIXTYPE", "HEALPIX",
            "HEALPix Pixelisation", status);
    fits_write_key(fptr, TSTRING, "ORDERING", order,
            "Pixel ordering scheme", status);
    fits_write_key(fptr, TLONG, "NSIDE", &nside,
            "Resolution parameter for HEALPix", status);
    fits_write_key(fptr, TSTRING, "COORDSYS", coord,
            "Pixelisation coordinate system", status);
    fits_write_comment(fptr,
            "G = Galactic, E = ecliptic, C = celestial = equatorial", status);
    fits_write_col(fptr, oskar_mem_is_double(data) ? TDOUBLE : TFLOAT,
            1, 1, 1, 12L * nside * nside, oskar_mem_void(data), status);
    fits_close_file(fptr, status);
}

#ifdef __cplusplus
}
#endif
