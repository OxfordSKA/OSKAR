/*
 * Copyright (c) 2017, The University of Oxford
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
#include "math/oskar_cmath.h"
#include "utility/oskar_file_exists.h"
#include <stdio.h>
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


static void write_pixels(oskar_Mem* data, const char* filename,
        int width, int height, int num_planes, int i_plane, int* status)
{
    long naxes[3], firstpix[3], num_pix;
    int dims_ok = 0;
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
        fits_create_img(f, oskar_mem_is_double(data) ? DOUBLE_IMG : FLOAT_IMG,
                3, naxes, status);
    }
    if (*status || !f)
    {
        if (f) fits_close_file(f, status);
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
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
    size_t i, num_elements;
    if (*status) return;
    num_elements = oskar_mem_length(input);
    if (oskar_mem_precision(input) == OSKAR_SINGLE)
    {
        const float *in;
        float *out;
        in = oskar_mem_float_const(input, status);
        out = oskar_mem_float(output, status);
        for (i = 0; i < num_elements; ++i) out[i] = in[2*i + offset];
    }
    else
    {
        const double *in;
        double *out;
        in = oskar_mem_double_const(input, status);
        out = oskar_mem_double(output, status);
        for (i = 0; i < num_elements; ++i) out[i] = in[2*i + offset];
    }
}


void oskar_mem_write_fits_cube(oskar_Mem* data, const char* root_name,
        int width, int height, int num_planes, int i_plane, int* status)
{
    oskar_Mem *copy = 0, *ptr = 0;
    size_t len, buf_len;
    char* fname;

    /* Checks. */
    if (*status) return;
    if (oskar_mem_is_matrix(data))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Construct the filename. */
    len = strlen(root_name);
    buf_len = 11 + len;
    fname = (char*) calloc(buf_len, sizeof(char));

    /* Copy to host memory if necessary. */
    ptr = data;
    if (oskar_mem_location(data) != OSKAR_CPU)
    {
        copy = oskar_mem_create_copy(ptr, OSKAR_CPU, status);
        ptr = copy;
    }

    /* Deal with complex data. */
    if (oskar_mem_is_complex(ptr))
    {
        oskar_Mem *temp;
        temp = oskar_mem_create(oskar_mem_precision(ptr), OSKAR_CPU,
                oskar_mem_length(ptr), status);

        /* Extract the real part and write it. */
        SNPRINTF(fname, buf_len, "%s_REAL.fits", root_name);
        convert_complex(ptr, temp, 0, status);
        write_pixels(temp, fname, width, height, num_planes, i_plane, status);

        /* Extract the imaginary part and write it. */
        SNPRINTF(fname, buf_len, "%s_IMAG.fits", root_name);
        convert_complex(ptr, temp, 1, status);
        write_pixels(temp, fname, width, height, num_planes, i_plane, status);
        oskar_mem_free(temp, status);
    }
    else
    {
        /* No conversion needed. */
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
        write_pixels(ptr, fname, width, height, num_planes, i_plane, status);
    }
    free(fname);
    oskar_mem_free(copy, status);
}

#ifdef __cplusplus
}
#endif
