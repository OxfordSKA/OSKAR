/*
 * Copyright (c) 2016-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_FROM_IMAGE_H_
#define OSKAR_SKY_FROM_IMAGE_H_

/**
 * @file oskar_sky_from_image.h
 */

#include "oskar_global.h"

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
oskar_Sky* oskar_sky_from_image(
        int precision,
        const oskar_Mem* image,
        const int image_size[2],
        const double image_crval_deg[2],
        const double image_crpix[2],
        double image_cellsize_deg,
        double image_freq_hz,
        double spectral_index,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
