/*
 * Copyright (c) 2013, The University of Oxford
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

#include <fits/oskar_fits_healpix_write_image.h>
#include <oskar_Image.h>
#include <oskar_mem.h>
#include <fitsio.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_fits_healpix_write_image(const char* filename, oskar_Image* image,
        int* status)
{
    fitsfile* fptr = 0;
    int type, c, channel_offset;
    int hdutype;
    long naxis[] = {0, 0};
    char order[] = { "RING" }; /* HEALPix ordering */
    char* ttype[] = { "SIGNAL" };
    char* tform[] = { "1E" };
    char* tunit[] = { " " };
    char coordsys[] = { "C" };
    long nside, npixels;

    /*
     * image.num_channels;
     * image.num_times;
     * image.num_pols;
     *
     * channels x times x pols x pixels
     * channel offset = c x num_times x num_pols x num_pixels
     */

    remove(filename);
    printf("A STATUS = %i\n", *status);
    fits_create_file(&fptr, filename, status);
    printf("B STATUS = %i\n", *status);
    fits_create_img(fptr, SHORT_IMG,  0, naxis, status);
    printf("C STATUS = %i\n", *status);
    fits_write_date(fptr, status);
    printf("D STATUS = %i\n", *status);

    /* FIXME this idea doesn't work with ds9 :-( */
    for (c = 0; c < image->num_channels; ++c)
    {
        printf("E STATUS = %i\n", *status);
        fits_movabs_hdu(fptr, c+1, &hdutype, status);
        nside = image->healpix_nside;
        npixels = 12L * nside * nside;
        fits_create_tbl(fptr, BINARY_TBL, npixels, 1, ttype, tform, tunit,
                "BINTABLE", status);
        fits_write_key(fptr, TSTRING, "PIXTYPE", "HEALPIX", "HEALPIX Pixelisation",
                status);
        fits_write_key(fptr, TSTRING, "ORDERING", order, "Pixel ordering scheme",
                status);
        fits_write_key(fptr, TLONG, "NSIDE", &nside,
                "Resolution parameter for HEALPIX", status);
        fits_write_key(fptr, TSTRING, "COORDSYS", coordsys,
                "Pixelisation coordinate system", status);
        fits_write_comment(fptr,
                "G = Galactic, E = ecliptic, C = celestial = equatorial", status);

        type = oskar_mem_type(&image->data);
        channel_offset = c * image->num_times * image->num_pols * npixels;
        if (type == OSKAR_DOUBLE)
        {
            double* data = oskar_mem_double(&image->data, status);
            fits_write_col(fptr, TDOUBLE, 1, 1, 1, npixels,
                    (void*)&data[channel_offset], status);
        }
        else if (type == OSKAR_SINGLE)
        {
            float* data = oskar_mem_float(&image->data, status);
            fits_write_col(fptr, TFLOAT, 1, 1, 1, npixels,
                    (void*)&data[channel_offset], status);
        }
    }
#if 0
    printf("STATUS = %i\n", *status);
    int num_hdu = 0;
    printf("XXXXXXXXXX %i\n", fits_get_num_hdus(fptr, &num_hdu, status));
    printf("XXXXXXXXXX %i\n", num_hdu);
    printf("STATUS = %i\n", *status);
#endif
    fits_close_file(fptr, status);
}


#ifdef __cplusplus
}
#endif
