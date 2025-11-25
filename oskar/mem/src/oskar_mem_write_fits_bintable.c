/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include <fitsio.h>

#include "mem/oskar_mem.h"
#include "utility/oskar_file_exists.h"

#ifdef __cplusplus
extern "C" {
#endif


void oskar_mem_write_fits_bintable(
        const char* file_name,
        const char* ext_name,
        unsigned int num_mem,
        size_t num_elements,
        int* status,
        ...
)
{
    va_list args;
    fitsfile* fptr = 0;
    oskar_Mem** handles = 0; /* Array of handles in CPU memory. */
    unsigned int i = 0;
    int i_hdu = 0, num_hdu = 0;
    char *ext_name_copy = 0, **tunit = 0, **ttype = 0, **tform = 0;
    if (*status || num_mem == 0) return;

    /* Check there are at least the number of specified elements in
     * each array. */
    va_start(args, status);
    for (i = 0; i < num_mem; ++i)
    {
        const oskar_Mem* mem = va_arg(args, const oskar_Mem*);
        (void) va_arg(args, const char*);
        if (oskar_mem_length(mem) < num_elements)
        {
            *status = OSKAR_ERR_DIMENSION_MISMATCH;
        }
    }
    va_end(args);
    if (*status) return;

    /* Allocate field headers. */
    const size_t ext_name_len = strlen(ext_name);
    ext_name_copy = (char*) calloc(1 + ext_name_len, sizeof(char));
    memcpy(ext_name_copy, ext_name, ext_name_len);
    tunit = (char**) calloc(num_mem, sizeof(char*));
    ttype = (char**) calloc(num_mem, sizeof(char*));
    tform = (char**) calloc(num_mem, sizeof(char*));

    /* Allocate and set up the column data. */
    handles = (oskar_Mem**) calloc(num_mem, sizeof(oskar_Mem*));
    va_start(args, status);
    for (i = 0; i < num_mem; ++i)
    {
        oskar_Mem* mem = 0;
        if (*status) break;
        mem = va_arg(args, oskar_Mem*);
        if (oskar_mem_location(mem) != OSKAR_CPU)
        {
            /* Only create a copy if it isn't already in CPU memory. */
            handles[i] = oskar_mem_create_copy(mem, OSKAR_CPU, status);
        }
        else
        {
            /* Otherwise, just store the pointer. */
            handles[i] = mem;
        }
        ttype[i] = va_arg(args, char*); /* Column name. */
        switch (oskar_mem_type(mem))
        {
        case OSKAR_INT:
            tform[i] = "1J"; /* 32-bit signed integer. */
            break;
        case OSKAR_SINGLE:
            tform[i] = "1E"; /* 32-bit float. */
            break;
        case OSKAR_DOUBLE:
            tform[i] = "1D"; /* 64-bit float. */
            break;
        case OSKAR_SINGLE_COMPLEX:
            tform[i] = "1C"; /* 32-bit complex pair. */
            break;
        case OSKAR_DOUBLE_COMPLEX:
            tform[i] = "1M"; /* 64-bit complex pair. */
            break;
        default:                                          /* LCOV_EXCL_LINE */
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
            break;                                        /* LCOV_EXCL_LINE */
        }
    }
    va_end(args);
    if (*status)
    {
        goto cleanup;                                     /* LCOV_EXCL_LINE */
    }

    /* Open the FITS file for writing. */
    if (oskar_file_exists(file_name))
    {
        fits_open_file(&fptr, file_name, READWRITE, status);
    }
    else
    {
        fits_create_file(&fptr, file_name, status);
    }
    if (*status || !fptr)
    {
        if (fptr) fits_close_file(fptr, status);          /* LCOV_EXCL_LINE */
        *status = OSKAR_ERR_FILE_IO;                      /* LCOV_EXCL_LINE */
        goto cleanup;                                     /* LCOV_EXCL_LINE */
    }

    /* Get number of HDUs in file. */
    fits_get_num_hdus(fptr, &num_hdu, status);

    /* Loop over extensions until the right binary table is found. */
    if (ext_name)
    {
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
            if (!strcmp(card1, ext_name))
            {
                break;
            }
        }
    }
    else
    {
        i_hdu = num_hdu + 1;
    }

    /* Create the table if it doesn't exist. */
    if (i_hdu > num_hdu)
    {
        fits_create_tbl(
                fptr, BINARY_TBL, (long int) num_elements,
                (int) num_mem, ttype, tform, tunit, ext_name_copy, status
        );
    }
    for (i = 0; i < num_mem; ++i)
    {
        int col_type = -1;
        switch (oskar_mem_type(handles[i]))
        {
        case OSKAR_INT:
            col_type = TINT;
            break;
        case OSKAR_SINGLE:
            col_type = TFLOAT;
            break;
        case OSKAR_DOUBLE:
            col_type = TDOUBLE;
            break;
        case OSKAR_SINGLE_COMPLEX:
            col_type = TCOMPLEX;
            break;
        case OSKAR_DOUBLE_COMPLEX:
            col_type = TDBLCOMPLEX;
            break;
        default:                                          /* LCOV_EXCL_LINE */
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
            break;                                        /* LCOV_EXCL_LINE */
        }
        fits_write_col(
                fptr, col_type, 1 + i, 1, 1, (long int) num_elements,
                oskar_mem_void(handles[i]), status
        );
    }
    fits_close_file(fptr, status);

    /* Free any temporary memory used by this function. */
    va_start(args, status);
    for (i = 0; i < num_mem; ++i)
    {
        const oskar_Mem* mem = va_arg(args, const oskar_Mem*);
        if (oskar_mem_location(mem) != OSKAR_CPU)
        {
            /* Only call free if we have made a copy here! */
            oskar_mem_free(handles[i], status);
        }
        (void) va_arg(args, const char*);
    }
    va_end(args);

cleanup:
    free(handles);
    free(tform);
    free(ttype);
    free(tunit);
    free(ext_name_copy);
}

#ifdef __cplusplus
}
#endif
