/*
 * Copyright (c) 2016-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <fitsio.h>

#include "log/oskar_log.h"
#include "mem/oskar_mem.h"
#include "utility/oskar_file_exists.h"

#ifdef __cplusplus
extern "C" {
#endif


void oskar_mem_write_healpix_fits(
        oskar_Mem* data,
        const char* filename,
        int overwrite,
        int nside,
        char ordering,
        char coordsys,
        int* status
)
{
    fitsfile* fptr = 0;
    char* ttype[] = { "SIGNAL" };
    char* tform[] = { "1?" };
    char coord[]  = { 0, 0 };
    if (*status) return;

    /* Check dimensions. */
    if ((int) oskar_mem_length(data) < 12 * nside * nside)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;           /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }

    /* Currently only "RING" ordering is supported. */
    if (ordering != 'R' && ordering != 'r')
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;             /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }

    /* Set the binary table column type code and the coordinate system. */
    tform[0] = oskar_mem_is_double(data) ? "1D" : "1E";
    coord[0] = coordsys;

    /* Open or create the file as needed. */
    if (oskar_file_exists(filename))
    {
        if (overwrite)
        {
            (void) remove(filename);
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
        oskar_log_error(                                  /* LCOV_EXCL_LINE */
                0, "Error creating HEALPix FITS file "    /* LCOV_EXCL_LINE */
                "(CFITSIO error code %d)", *status        /* LCOV_EXCL_LINE */
        );
        *status = OSKAR_ERR_FILE_IO;                      /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }

    /* Write the data into a new binary table. */
    fits_create_tbl(
            fptr, BINARY_TBL, 12L * nside * nside, 1,
            ttype, tform, NULL, "BINTABLE", status
    );
    fits_write_key(
            fptr, TSTRING, "PIXTYPE", "HEALPIX",
            "HEALPix Pixelisation", status
    );
    fits_write_key(
            fptr, TSTRING, "ORDERING", "RING",
            "Pixel ordering scheme", status
    );
    fits_write_key(
            fptr, TINT, "NSIDE", &nside,
            "Resolution parameter for HEALPix", status
    );
    fits_write_key(
            fptr, TSTRING, "COORDSYS", coord,
            "Pixelisation coordinate system", status
    );
    fits_write_comment(
            fptr,
            "G = Galactic, E = ecliptic, C = celestial = equatorial", status
    );
    fits_write_col(
            fptr, oskar_mem_is_double(data) ? TDOUBLE : TFLOAT,
            1, 1, 1, 12L * nside * nside, oskar_mem_void(data), status
    );
    fits_close_file(fptr, status);
}

#ifdef __cplusplus
}
#endif
