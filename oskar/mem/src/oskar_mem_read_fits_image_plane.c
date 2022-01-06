/*
 * Copyright (c) 2016-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"
#include "math/oskar_cmath.h"
#include <fitsio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_AXES 10
#define FACTOR (2.0*sqrt(2.0*log(2.0)))

oskar_Mem* oskar_mem_read_fits_image_plane(const char* filename, int i_time,
        int i_chan, int i_stokes, int* image_size, double* image_crval_deg,
        double* image_crpix, double* image_cellsize_deg,
        double* image_time, double* image_freq_hz, double* beam_area_pixels,
        char** brightness_units, int* status)
{
    int i = 0, naxis = 0, imagetype = 0, anynul = 0;
    int status1 = 0, status2 = 0, type_fits = 0, type_oskar = 0;
    int axis_time = -1, axis_chan = -1, axis_stokes = -1;
    long num_pixels = 0, naxes[MAX_AXES], firstpix[MAX_AXES];
    char card[FLEN_CARD], *ctype[MAX_AXES], ctype_str[MAX_AXES][FLEN_VALUE];
    double crval[MAX_AXES], crpix[MAX_AXES], cdelt[MAX_AXES];
    double nul = 0.0, bmaj = 0.0, bmin = 0.0;
    fitsfile* fptr = 0;
    oskar_Mem* data = 0;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Open the file. */
    fits_open_file(&fptr, filename, READONLY, status);
    if (*status || !fptr)
    {
        *status = OSKAR_ERR_FILE_IO;
        return 0;
    }

    /* Get the image parameters. */
    fits_get_img_param(fptr, MAX_AXES, &imagetype, &naxis, naxes, status);
    if (*status) goto file_error;

    /* Set the data type. */
    if (imagetype != FLOAT_IMG && imagetype != DOUBLE_IMG) goto file_error;
    type_fits  = (imagetype == FLOAT_IMG) ? TFLOAT : TDOUBLE;
    type_oskar = (imagetype == FLOAT_IMG) ? OSKAR_SINGLE : OSKAR_DOUBLE;

    /* Check that the FITS image contains at least two dimensions. */
    if (naxis < 2 || naxis > MAX_AXES) goto file_error;
    num_pixels = naxes[0] * naxes[1];

    /* Read all CTYPE, CDELT, CRPIX, CRVAL values, ignoring errors. */
    for (i = 0; i < MAX_AXES; ++i)
    {
        ctype[i] = ctype_str[i];
        cdelt[i] = 0.0;
        crpix[i] = 0.0;
        crval[i] = 0.0;
        firstpix[i] = 1;
    }
    *status = 0;
    fits_read_keys_str(fptr, "CTYPE", 1, naxis, ctype, &i, status);
    *status = 0;
    fits_read_keys_dbl(fptr, "CRPIX", 1, naxis, crpix, &i, status);
    *status = 0;
    fits_read_keys_dbl(fptr, "CRVAL", 1, naxis, crval, &i, status);
    *status = 0;
    fits_read_keys_dbl(fptr, "CDELT", 1, naxis, cdelt, &i, status);
    if (cdelt[0] == 0.0 || cdelt[1] == 0.0)
    {
        double cd1_1 = 0.0, cd1_2 = 0.0, cd2_1 = 0.0, cd2_2 = 0.0;
        *status = 0;
        fits_read_key(fptr, TDOUBLE, "CD1_1", &cd1_1, 0, status);
        fits_read_key(fptr, TDOUBLE, "CD1_2", &cd1_2, 0, status);
        fits_read_key(fptr, TDOUBLE, "CD2_1", &cd2_1, 0, status);
        fits_read_key(fptr, TDOUBLE, "CD2_2", &cd2_2, 0, status);
        if (cd1_2 == 0.0 && cd2_1 == 0.0 && !*status)
        {
            /* Accept CD matrix values as long as the matrix is diagonal. */
            cdelt[0] = cd1_1;
            cdelt[1] = cd2_2;
        }
        *status = 0;
    }

    /* Identify the axes. */
    for (i = 0; i < naxis; ++i)
    {
        if (strncmp(ctype[i], "STOKES", 6) == 0) {
            axis_stokes = i;
        } else if (strncmp(ctype[i], "FREQ", 4) == 0) {
            axis_chan = i;
        } else if (strncmp(ctype[i], "TIME", 4) == 0) {
            axis_time = i;
        }
    }

    /* Check ranges and set the dimensions to read. */
    if (axis_stokes >= 0)
    {
        if (i_stokes >= naxes[axis_stokes])
        {
            goto range_error;
        }
        firstpix[axis_stokes] = 1 + i_stokes;
    }
    else if (i_stokes > 0)
    {
        goto range_error;
    }
    if (axis_chan >= 0)
    {
        if (i_chan >= naxes[axis_chan])
        {
            goto range_error;
        }
        firstpix[axis_chan] = 1 + i_chan;
    }
    else if (i_chan > 0)
    {
        goto range_error;
    }
    if (axis_time >= 0)
    {
        if (i_time >= naxes[axis_time])
        {
            goto range_error;
        }
        firstpix[axis_time] = 1 + i_time;
    }
    else if (i_time > 0)
    {
        goto range_error;
    }

    /* Return requested image metadata. */
    if (image_size)
    {
        image_size[0] = naxes[0];
        image_size[1] = naxes[1];
    }
    if (image_crval_deg)
    {
        image_crval_deg[0] = crval[0];
        image_crval_deg[1] = crval[1];
    }
    if (image_crpix)
    {
        image_crpix[0] = crpix[0];
        image_crpix[1] = crpix[1];
    }
    if (image_cellsize_deg)
    {
        *image_cellsize_deg = fabs(cdelt[1]);
    }
    if (image_freq_hz && axis_chan >= 0)
    {
        *image_freq_hz = crval[axis_chan] + i_chan * cdelt[axis_chan];
    }
    if (image_time && axis_time >= 0)
    {
        *image_time    = crval[axis_time] + i_time * cdelt[axis_time];
    }

    /* Search for beam size in header keywords first. */
    status1 = status2 = 0;
    fits_read_key(fptr, TDOUBLE, "BMAJ", &bmaj, 0, &status1);
    fits_read_key(fptr, TDOUBLE, "BMIN", &bmin, 0, &status2);
    if (status1 || status2)
    {
        int cards = 0;

        /* If header keywords don't exist, search all the history cards. */
        fits_get_hdrspace(fptr, &cards, 0, status);
        if (*status) goto file_error;
        for (i = 0; i < cards; ++i)
        {
            fits_read_record(fptr, i, card, status);
            if (!strncmp(card, "HISTORY AIPS   CLEAN BMAJ", 25))
            {
                bmaj = strtod(card + 26, 0);
                bmin = strtod(card + 44, 0);
                break;
            }
        }
    }

    /* Calculate beam area. */
    if (beam_area_pixels && fabs(cdelt[0]) > 0.0)
    {
        *beam_area_pixels = 2.0 * M_PI * (bmaj * bmin)
                        / (FACTOR * FACTOR * cdelt[0] * cdelt[0]);
    }

    /* Get brightness units if present. */
    status1 = 0;
    fits_read_key(fptr, TSTRING, "BUNIT", card, 0, &status1);
    i = (int) strlen(card);
    if (!status1 && i > 0 && brightness_units)
    {
        *brightness_units = (char*) realloc (*brightness_units, i + 1);
        memcpy(*brightness_units, card, i + 1);
    }

    /* Read image pixel data. */
    data = oskar_mem_create(type_oskar, OSKAR_CPU, num_pixels, status);
    fits_read_pix(fptr, type_fits, firstpix, num_pixels,
            &nul, oskar_mem_void(data), &anynul, status);
    fits_close_file(fptr, status);
    return data;

    /* Error conditions. */
range_error:
    oskar_mem_free(data, status);
    fits_close_file(fptr, status);
    *status = OSKAR_ERR_OUT_OF_RANGE;
    return 0;

file_error:
    oskar_mem_free(data, status);
    fits_close_file(fptr, status);
    *status = OSKAR_ERR_FILE_IO;
    return 0;
}

#ifdef __cplusplus
}
#endif
