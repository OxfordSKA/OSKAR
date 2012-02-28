/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_FITS_WRITE_IMAGE_H_
#define OSKAR_FITS_WRITE_IMAGE_H_

/**
 * @file oskar_fits_write_image.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Writes a block of memory to a simple 2D FITS image.
 *
 * @details
 * This function writes the contents of a block of memory to a FITS image.
 * The supplied parameters are written into the FITS header.
 *
 * This function is intended to be used for simple 2D images of one frequency
 * channel and one polarisation (Stokes I) only.
 *
 * Note that astronomical FITS images have their first pixel at the
 * BOTTOM LEFT of the image, and their last pixel at the TOP RIGHT
 * of the image.
 *
 * Although the FITS documentation says that "the ordering of arrays in
 * FITS files ... is more similar to the dimensionality of arrays in Fortran
 * rather than C" (http://heasarc.gsfc.nasa.gov/fitsio/c/c_user/node75.html),
 * confusingly, the fastest-varying dimension is across the columns - i.e.
 * C-ordered!
 *
 * By convention, astronomical radio FITS images are written with the
 * following axis parameter keywords:
 *
 * - CTYPE1 = 'RA---SIN'
 * - CDELT1 = (negative value)
 * - CTYPE2 = 'DEC--SIN'
 * - CDELT2 = (positive value)
 *
 * The first pixel therefore corresponds to:
 * - the LARGEST Right Ascension, and
 * - the SMALLEST Declination.
 *
 * The last pixel corresponds to:
 * - the SMALLEST Right Ascension, and
 * - the LARGEST Declination.
 *
 * The fastest varying dimension is along the RA axis.
 *
 */
OSKAR_EXPORT
void oskar_fits_write_image(const char* filename, int type, int width,
        int height, void* data, double ra, double dec, double pixel_scale,
        double frequency, double bandwidth);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_FITS_WRITE_IMAGE_H_ */
