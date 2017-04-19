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

#ifndef OSKAR_SKY_FROM_IMAGE_H_
#define OSKAR_SKY_FROM_IMAGE_H_

/**
 * @file oskar_sky_from_image.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Creates a sky model from an array of image pixels.
 *
 * @details
 * Creates a sky model from an array of image pixels.
 *
 * @param[in] precision           Enumerated precision of the output sky model.
 * @param[in] image               2D image data (ordered as in FITS image).
 * @param[in] image_size          Image size[2] (width and height).
 * @param[in] image_crval_deg     Image centre coordinates[2], in degrees.
 * @param[in] image_crpix         Image centre pixels[2] (1-based).
 * @param[in] image_cellsize_deg  Image pixel size, in degrees.
 * @param[in] image_freq_hz       Frequency value of the plane, in Hz.
 * @param[in] spectral_index      Spectral index value of pixels.
 * @param[in,out] status          Status return code.
 */
OSKAR_EXPORT
oskar_Sky* oskar_sky_from_image(int precision, const oskar_Mem* image,
        const int image_size[2], const double image_crval_deg[2],
        const double image_crpix[2], double image_cellsize_deg,
        double image_freq_hz, double spectral_index, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_FROM_IMAGE_H_ */
