/*
 * Copyright (c) 2017-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"
#include "math/oskar_cmath.h"
#include "utility/oskar_file_exists.h"
#include <stdio.h>
#include <stdlib.h>
#include <fitsio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif


#if __STDC_VERSION__ >= 199901L
#define SNPRINTF(BUF, SIZE, FMT, ...) snprintf(BUF, SIZE, FMT, __VA_ARGS__);
#else
#define SNPRINTF(BUF, SIZE, FMT, ...) sprintf(BUF, FMT, __VA_ARGS__);
#endif


static void write_pixels(oskar_Mem* data, const char* filename, int i_hdu,
        int width, int height, int num_planes, int i_plane, int* status)
{
    long naxes[3], firstpix[3], num_pix = 0;
    int dims_ok = 0, num_hdus = 0;
    fitsfile* f = 0;
    if (*status) return;
    if (oskar_file_exists(filename))
    {
        int naxis = 0, imagetype = 0;
        fits_open_file(&f, filename, READWRITE, status);
        fits_get_img_param(f, 3, &imagetype, &naxis, naxes, status);
        if (naxis == 3 &&
                naxes[0] == width &&
                naxes[1] == height &&
                naxes[2] == num_planes)
        {
            dims_ok = 1;
        }
        else
        {
            *status = 0;
            fits_close_file(f, status);
            remove(filename);
            f = 0;
        }
    }
    if (!dims_ok)
    {
        naxes[0] = width;
        naxes[1] = height;
        naxes[2] = num_planes;
        fits_create_file(&f, filename, status);
    }
    if (*status || !f)
    {
        if (f) fits_close_file(f, status);
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
    fits_get_num_hdus(f, &num_hdus, status);
    if (i_hdu >= num_hdus)
    {
        fits_create_img(f, oskar_mem_is_double(data) ? DOUBLE_IMG : FLOAT_IMG,
                3, naxes, status);
    }
    fits_movabs_hdu(f, i_hdu + 1, NULL, status);
    num_pix = width * height;
    firstpix[0] = 1;
    firstpix[1] = 1;
    firstpix[2] = 1 + i_plane;
    if (i_plane < 0)
    {
        firstpix[2] = 1;
        num_pix *= num_planes;
    }
    fits_write_pix(f, oskar_mem_is_double(data) ? TDOUBLE : TFLOAT,
            firstpix, num_pix, oskar_mem_void(data), status);
    fits_close_file(f, status);
}


static void convert_complex(const oskar_Mem* input, oskar_Mem* output,
        int offset, int* status)
{
    size_t i = 0, num_elements = 0;
    if (*status) return;
    num_elements = oskar_mem_length(input);
    if (oskar_mem_precision(input) == OSKAR_SINGLE)
    {
        const float *in = 0;
        float *out = 0;
        in = oskar_mem_float_const(input, status);
        out = oskar_mem_float(output, status);
        for (i = 0; i < num_elements; ++i) out[i] = in[2*i + offset];
    }
    else
    {
        const double *in = 0;
        double *out = 0;
        in = oskar_mem_double_const(input, status);
        out = oskar_mem_double(output, status);
        for (i = 0; i < num_elements; ++i) out[i] = in[2*i + offset];
    }
}


void oskar_mem_write_fits_cube(oskar_Mem* data, const char* root_name,
        int width, int height, int num_planes, int i_plane, int* status)
{
    oskar_Mem *copy = 0, *ptr = 0;
    char* fname = 0;
    if (*status) return;
    if (oskar_mem_is_matrix(data))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Construct the filename. */
    const size_t len = strlen(root_name);
    const size_t buf_len = 11 + len;
    fname = (char*) calloc(buf_len, sizeof(char));

    /* Copy to host memory if necessary. */
    ptr = data;
    if (oskar_mem_location(data) != OSKAR_CPU)
    {
        copy = oskar_mem_create_copy(ptr, OSKAR_CPU, status);
        ptr = copy;
    }

    /* Check filename extension. */
    if ((len >= 5) && (
            !strcmp(&(root_name[len-5]), ".fits") ||
            !strcmp(&(root_name[len-5]), ".FITS") ))
    {
        SNPRINTF(fname, buf_len, "%s", root_name);
    }
    else
    {
        SNPRINTF(fname, buf_len, "%s.fits", root_name);
    }

    /* Deal with complex data. */
    if (oskar_mem_is_complex(ptr))
    {
        oskar_Mem *temp = 0;
        temp = oskar_mem_create(oskar_mem_precision(ptr), OSKAR_CPU,
                oskar_mem_length(ptr), status);

        /* Extract the real part and write it to the first HDU. */
        convert_complex(ptr, temp, 0, status);
        write_pixels(temp, fname, 0,
                width, height, num_planes, i_plane, status);

        /* Extract the imaginary part and write it to the second HDU. */
        convert_complex(ptr, temp, 1, status);
        write_pixels(temp, fname, 1,
                width, height, num_planes, i_plane, status);
        oskar_mem_free(temp, status);
    }
    else
    {
        /* No conversion needed. */
        write_pixels(ptr, fname, 0,
                width, height, num_planes, i_plane, status);
    }
    free(fname);
    oskar_mem_free(copy, status);
}

#ifdef __cplusplus
}
#endif
