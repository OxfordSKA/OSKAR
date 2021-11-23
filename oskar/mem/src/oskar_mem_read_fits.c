/*
 * Copyright (c) 2019-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"
#include <stdio.h>
#include <stdlib.h>
#include <fitsio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_AXES 10

void oskar_mem_read_fits(oskar_Mem* data, size_t offset, size_t num_pixels,
        const char* file_name, int num_index_dims, const int* start_index,
        int* num_axes, int** axis_size, double** axis_inc, int* status)
{
    int i = 0, imagetype = 0, naxis = 0, anynul = 0, type_fits = 0;
    long firstpix[MAX_AXES], naxes[MAX_AXES], num_pixels_cube = 1;
    double cdelt[MAX_AXES];
    double nul = 0.0;
    oskar_Mem *data_ptr = 0, *data_temp = 0;
    fitsfile* fptr = 0;
    if (*status) return;

    /* Open the file and get the cube parameters. */
    fits_open_file(&fptr, file_name, READONLY, status);
    if (*status || !fptr)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
    fits_get_img_param(fptr, MAX_AXES, &imagetype, &naxis, naxes, status);
    if (*status || naxis < 1 || naxis > MAX_AXES)
    {
        fits_close_file(fptr, status);
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Store axis sizes and increments. */
    if (axis_inc)
    {
        fits_read_keys_dbl(fptr, "CDELT", 1, naxis, cdelt, &i, status);
    }
    if (num_axes && *num_axes < naxis)
    {
        if (axis_size)
        {
            *axis_size = (int*) realloc(*axis_size, naxis * sizeof(int));
        }
        if (axis_inc)
        {
            *axis_inc = (double*) realloc(*axis_inc, naxis * sizeof(double));
        }
    }
    if (num_axes) *num_axes = naxis;
    for (i = 0; i < naxis; ++i)
    {
        firstpix[i] = 1;
        if (axis_size) (*axis_size)[i] = naxes[i];
        if (axis_inc) (*axis_inc)[i] = cdelt[i];
        num_pixels_cube *= naxes[i];
    }
    if (!data || num_index_dims == 0 || !start_index)
    {
        fits_close_file(fptr, status);
        return;
    }

    /* Get the FITS data type of the output array. */
    switch (oskar_mem_type(data))
    {
    case OSKAR_INT:
        type_fits = TINT;
        break;
    case OSKAR_SINGLE:
        type_fits = TFLOAT;
        break;
    case OSKAR_DOUBLE:
        type_fits = TDOUBLE;
        break;
    default:
        fits_close_file(fptr, status);
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Read the data. */
    offset *= oskar_mem_element_size(oskar_mem_type(data));
    if (num_index_dims > 0)
    {
        for (i = 0; i < naxis && i < num_index_dims; ++i)
        {
            firstpix[i] = 1 + start_index[i];
        }
    }
    else if (num_pixels == 0)
    {
        num_pixels = (size_t) num_pixels_cube;
    }
    data_ptr = data;
    if (oskar_mem_location(data) != OSKAR_CPU)
    {
        data_temp = oskar_mem_create(
                oskar_mem_type(data), OSKAR_CPU, num_pixels, status);
        data_ptr = data_temp;
    }
    oskar_mem_ensure(data_ptr, num_pixels, status);
    fits_read_pix(fptr, type_fits, firstpix, (long) num_pixels,
            &nul, oskar_mem_char(data_ptr) + offset, &anynul, status);
    fits_close_file(fptr, status);
    if (data_ptr != data)
    {
        oskar_mem_copy(data, data_ptr, status);
    }
    oskar_mem_free(data_temp, status);
}

#ifdef __cplusplus
}
#endif
