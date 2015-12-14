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

#ifndef OSKAR_FITS_IMAGE_TO_SKY_MODEL_H_
#define OSKAR_FITS_IMAGE_TO_SKY_MODEL_H_

/**
 * @file oskar_fits_image_to_sky_model.h
 */

#include <oskar_global.h>
#include <oskar_sky.h>
#include <oskar_log.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Reads a FITS file into a sky model structure.
 *
 * @details
 * This function reads data from a FITS file into an OSKAR sky model structure.
 *
 * @param[in,out] ptr    Pointer to log structure to use.
 * @param[in] filename   File name of FITS image to read.
 * @param[out] sky       Pointer to sky model to fill.
 * @param[in] spectral_index Spectral index for all output pixels.
 * @param[in] min_peak_fraction Minimum allowed fraction of image peak.
 * @param[in] noise_floor Ignore pixels below this value (units are Jy/PIXEL).
 * @param[in] downsample_factor Factor by which to downsample the image before
 *                              performing the format conversion.
 *
 * @return An error code.
 */
OSKAR_FITS_EXPORT
int oskar_fits_image_to_sky_model(oskar_Log* ptr, const char* filename,
        oskar_Sky* sky, double spectral_index, double min_peak_fraction,
        double noise_floor, int downsample_factor);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_FITS_IMAGE_TO_SKY_MODEL_H_ */
