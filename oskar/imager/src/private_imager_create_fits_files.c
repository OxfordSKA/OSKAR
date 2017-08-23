/*
 * Copyright (c) 2016-2017, The University of Oxford
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

#include "imager/private_imager.h"

#include "math/oskar_cmath.h"
#include "convert/oskar_convert_fov_to_cellsize.h"
#include "imager/oskar_imager.h"
#include "imager/private_imager_create_fits_files.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <fitsio.h>

#if __STDC_VERSION__ >= 199901L
#define SNPRINTF(BUF, SIZE, FMT, ...) snprintf(BUF, SIZE, FMT, __VA_ARGS__);
#else
#define SNPRINTF(BUF, SIZE, FMT, ...) sprintf(BUF, FMT, __VA_ARGS__);
#endif

#ifdef __cplusplus
extern "C" {
#endif

static fitsfile* create_fits_file(const char* filename, int precision,
        int width, int height, int num_channels, double centre_deg[2],
        double fov_deg[2], double start_freq_hz, double delta_freq_hz,
        int* status);
static void write_axis_header(fitsfile* fptr, int axis_id,
        const char* ctype, const char* ctype_comment, double crval,
        double cdelt, double crpix, double crota, int* status);

void oskar_imager_create_fits_files(oskar_Imager* h, int* status)
{
    int i;
    if (*status) return;
    if (!h->output_root) return;
    oskar_timer_resume(h->tmr_write);
    for (i = 0; i < h->num_im_pols; ++i)
    {
        double fov_deg[2];
        char f[FILENAME_MAX];
        const char *a[] = {"I","Q","U","V"}, *b[] = {"XX","XY","YX","YY"};

        /* Construct filename based on image type. */
        switch (h->im_type)
        {
        case OSKAR_IMAGE_TYPE_STOKES:
            SNPRINTF(f, sizeof(f), "%s_%s.fits", h->output_root, a[i]); break;
        case OSKAR_IMAGE_TYPE_I:
            SNPRINTF(f, sizeof(f), "%s_I.fits", h->output_root); break;
        case OSKAR_IMAGE_TYPE_Q:
            SNPRINTF(f, sizeof(f), "%s_Q.fits", h->output_root); break;
        case OSKAR_IMAGE_TYPE_U:
            SNPRINTF(f, sizeof(f), "%s_U.fits", h->output_root); break;
        case OSKAR_IMAGE_TYPE_V:
            SNPRINTF(f, sizeof(f), "%s_V.fits", h->output_root); break;
        case OSKAR_IMAGE_TYPE_LINEAR:
            SNPRINTF(f, sizeof(f), "%s_%s.fits", h->output_root, b[i]); break;
        case OSKAR_IMAGE_TYPE_XX:
            SNPRINTF(f, sizeof(f), "%s_XX.fits", h->output_root); break;
        case OSKAR_IMAGE_TYPE_XY:
            SNPRINTF(f, sizeof(f), "%s_XY.fits", h->output_root); break;
        case OSKAR_IMAGE_TYPE_YX:
            SNPRINTF(f, sizeof(f), "%s_YX.fits", h->output_root); break;
        case OSKAR_IMAGE_TYPE_YY:
            SNPRINTF(f, sizeof(f), "%s_YY.fits", h->output_root); break;
        case OSKAR_IMAGE_TYPE_PSF:
            SNPRINTF(f, sizeof(f), "%s_PSF.fits", h->output_root); break;
        default:
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            break;
        }

        fov_deg[0] = fov_deg[1] = h->fov_deg;
        h->fits_file[i] = create_fits_file(f, h->imager_prec, h->image_size,
                h->image_size, h->num_im_channels, h->im_centre_deg, fov_deg,
                h->im_freqs[0], h->freq_inc_hz, status);
        h->output_name[i] = (char*) realloc(h->output_name[i], 1 + strlen(f));
        strcpy(h->output_name[i], f);
    }
    oskar_timer_pause(h->tmr_write);
}


fitsfile* create_fits_file(const char* filename, int precision,
        int width, int height, int num_channels, double centre_deg[2],
        double fov_deg[2], double start_freq_hz, double delta_freq_hz,
        int* status)
{
    long naxes[3];
    double delta;
    fitsfile* f = 0;
    FILE* t = 0;
    if (*status) return 0;

    /* Create a new FITS file and write the image headers. */
    t = fopen(filename, "rb");
    if (t)
    {
        fclose(t);
        remove(filename);
    }
    naxes[0]  = width;
    naxes[1]  = height;
    naxes[2]  = num_channels;
    fits_create_file(&f, filename, status);
    fits_create_img(f, (precision == OSKAR_DOUBLE ? DOUBLE_IMG : FLOAT_IMG),
            3, naxes, status);
    fits_set_hdrsize(f, 120, status); /* Reserve some header space for log. */
    fits_write_date(f, status);

    /* Write axis headers. */
    delta = oskar_convert_fov_to_cellsize(fov_deg[0] * M_PI/180, width) * 180/M_PI;
    write_axis_header(f, 1, "RA---SIN", "Right Ascension",
            centre_deg[0], -delta, width / 2 + 1, 0.0, status);
    delta = oskar_convert_fov_to_cellsize(fov_deg[1] * M_PI/180, height) * 180/M_PI;
    write_axis_header(f, 2, "DEC--SIN", "Declination",
            centre_deg[1], delta, height / 2 + 1, 0.0, status);
    write_axis_header(f, 3, "FREQ", "Frequency",
            start_freq_hz, delta_freq_hz, 1.0, 0.0, status);

    /* Write other headers. */
    fits_write_key_str(f, "BUNIT", "JY/BEAM", "Brightness units", status);
    fits_write_key_dbl(f, "OBSRA", centre_deg[0], 10, "RA", status);
    fits_write_key_dbl(f, "OBSDEC", centre_deg[1], 10, "DEC", status);
    /*fits_flush_file(f, status);*/

    return f;
}


void write_axis_header(fitsfile* fptr, int axis_id,
        const char* ctype, const char* ctype_comment, double crval,
        double cdelt, double crpix, double crota, int* status)
{
    char key[FLEN_KEYWORD], value[FLEN_VALUE], comment[FLEN_COMMENT];
    int decimals = 17;
    if (*status) return;
    strncpy(comment, ctype_comment, FLEN_COMMENT-1);
    strncpy(value, ctype, FLEN_VALUE-1);
    fits_make_keyn("CTYPE", axis_id, key, status);
    fits_write_key_str(fptr, key, value, comment, status);
    fits_make_keyn("CRVAL", axis_id, key, status);
    fits_write_key_dbl(fptr, key, crval, decimals, NULL, status);
    fits_make_keyn("CDELT", axis_id, key, status);
    fits_write_key_dbl(fptr, key, cdelt, decimals, NULL, status);
    fits_make_keyn("CRPIX", axis_id, key, status);
    fits_write_key_dbl(fptr, key, crpix, decimals, NULL, status);
    fits_make_keyn("CROTA", axis_id, key, status);
    fits_write_key_dbl(fptr, key, crota, decimals, NULL, status);
}

#ifdef __cplusplus
}
#endif
