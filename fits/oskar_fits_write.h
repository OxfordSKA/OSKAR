/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#ifndef OSKAR_FITS_WRITE_H_
#define OSKAR_FITS_WRITE_H_

/**
 * @file oskar_fits_write.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Writes a block of memory to a multidimensional FITS hyper-cube.
 *
 * @details
 * This function writes the contents of a block of memory to a
 * multidimensional FITS hyper-cube.
 * The supplied parameters are written into the FITS header.
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
 * @param[in] filename   File name of FITS image to write.
 * @param[in] type       Data type (OSKAR_SINGLE or OSKAR_DOUBLE) of the array.
 * @param[in] naxis      The number of axes in the hyper-cube.
 * @param[in] naxes      An array containing the length of each axis.
 * @param[in] data       Pointer to memory block to write.
 * @param[in] ctype      Array of CTYPE keyword values for each axis.
 * @param[in] ctype_desc Array of CTYPE keyword comments for each axis.
 * @param[in] crval      Array of CRVAL keyword (reference) values for each axis.
 * @param[in] cdelt      Array of CDELT keyword (delta) values for each axis.
 * @param[in] crpix      Array of CRPIX keyword (reference pixel) values for each axis.
 * @param[in] crpix      Array of CROTA keyword (rotation) values for each axis.
 * @param[in,out] status Status return code.
 */
OSKAR_FITS_EXPORT
void oskar_fits_write(const char* filename, int type, int naxis,
        long* naxes, void* data, const char** ctype, const char** ctype_desc,
        const double* crval, const double* cdelt, const double* crpix,
        const double* crota, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_FITS_WRITE_H_ */
