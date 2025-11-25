/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fitsio.h>

#include "log/oskar_log.h"
#include "mem/oskar_mem.h"

#ifdef __cplusplus
extern "C" {
#endif


oskar_Mem* oskar_mem_read_fits_bintable(
        const char* file_name,
        const char* ext_name,
        const char* column_name,
        int* status
)
{
    int col_index = 1, i_hdu = 0, num_hdu = 0, type_fits = 0, type_oskar = 0;
    long repeat = 0, width = 0, num_rows = 0;
    char* column_name_copy = 0;
    fitsfile* fptr = 0;
    oskar_Mem* data = 0;
    if (*status) return 0;

    /* Open the FITS file. */
    fits_open_file(&fptr, file_name, READONLY, status);
    if (*status || !fptr)
    {
        oskar_log_error(0, "Error opening FITS file '%s'", file_name);
        *status = OSKAR_ERR_FILE_IO;
        return 0;
    }

    /* Get number of HDUs in file. */
    fits_get_num_hdus(fptr, &num_hdu, status);

    /* Loop over extensions until the right binary table is found. */
    for (i_hdu = 1; i_hdu <= num_hdu; ++i_hdu)
    {
        char card1[FLEN_CARD];
        int hdutype = 0;
        fits_movabs_hdu(fptr, i_hdu, &hdutype, status);
        if (hdutype != BINARY_TBL)
        {
            continue;
        }
        fits_read_key(fptr, TSTRING, "EXTNAME", card1, NULL, status);
        if (!ext_name || !strcmp("", ext_name) || !strcmp(card1, ext_name))
        {
            break;
        }
    }

    /* Check if we exhausted the HDU list without finding anything. */
    if (i_hdu > num_hdu)
    {
        oskar_log_error(
                0, "Binary table '%s' not found in FITS file '%s'",
                column_name_copy, file_name
        );
        fits_close_file(fptr, status);
        *status = OSKAR_ERR_FILE_IO;
        return 0;
    }

    /* Get column index and number of rows. */
    const size_t column_name_len = strlen(column_name);
    column_name_copy = (char*) calloc(1 + column_name_len, 1);
    memcpy(column_name_copy, column_name, column_name_len);
    fits_get_colnum(fptr, CASEINSEN, column_name_copy, &col_index, status);
    if (*status == COL_NOT_FOUND)
    {
        oskar_log_error(
                0, "Column '%s' not found in FITS file '%s'",
                column_name_copy, file_name
        );
        fits_close_file(fptr, status);
        free(column_name_copy);
        *status = OSKAR_ERR_FILE_IO;
        return 0;
    }
    fits_get_num_rows(fptr, &num_rows, status);

    /* Get the column data type and check it. */
    fits_get_coltype(fptr, col_index, &type_fits, &repeat, &width, status);
    switch (type_fits)
    {
    case TINT:
    case TLONG:
        /* Data type reported as TLONG, but should be read as TINT... */
        /* It seems rather strange that we should have to do this! */
        /* Check the width as well. */
        if (width != sizeof(int))
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
        }
        else
        {
            type_fits = TINT;
            type_oskar = OSKAR_INT;
        }
        break;
    case TFLOAT:
        if (width != sizeof(float))
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
        }
        else
        {
            type_oskar = OSKAR_SINGLE;
        }
        break;
    case TDOUBLE:
        if (width != sizeof(double))
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
        }
        else
        {
            type_oskar = OSKAR_DOUBLE;
        }
        break;
    case TCOMPLEX:
        if (width != 2 * sizeof(float))
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
        }
        else
        {
            type_oskar = OSKAR_SINGLE_COMPLEX;
        }
        break;
    case TDBLCOMPLEX:
        if (width != 2 * sizeof(double))
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
        }
        else
        {
            type_oskar = OSKAR_DOUBLE_COMPLEX;
        }
        break;
    default:                                              /* LCOV_EXCL_LINE */
        *status = OSKAR_ERR_BAD_DATA_TYPE;                /* LCOV_EXCL_LINE */
        break;                                            /* LCOV_EXCL_LINE */
    }

    /* Read the column if no error detected. */
    if (!*status)
    {
        int anynul = 0;
        data = oskar_mem_create(type_oskar, OSKAR_CPU, num_rows, status);
        fits_read_col(
                fptr, type_fits, col_index, 1, 1, num_rows, 0,
                oskar_mem_void(data), &anynul, status
        );
    }
    fits_close_file(fptr, status);
    free(column_name_copy);
    return data;
}

#ifdef __cplusplus
}
#endif
