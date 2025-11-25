/*
 * Copyright (c) 2016-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_FROM_HEALPIX_RING_H_
#define OSKAR_SKY_FROM_HEALPIX_RING_H_

/**
 * @file oskar_sky_from_healpix_ring.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Creates a sky model from an array of HEALPix pixels.
 *
 * @details
 * Creates a sky model from an array of HEALPix pixels.
 * The pixellisation must be in RING format.
 *
 * @param[in] precision           Enumerated precision of the output sky model.
 * @param[in] data                HEALPix data array (values in Jy).
 * @param[in] frequency_hz        Reference frequency, in Hz.
 * @param[in] spectral_index      Spectral index to give to each pixel.
 * @param[in] nside               HEALPix resolution parameter.
 * @param[in] galactic_coords     If true, map is in Galactic coordinates;
 *                                otherwise, equatorial coordinates.
 * @param[in,out] status          Status return code.
 */
OSKAR_EXPORT
oskar_Sky* oskar_sky_from_healpix_ring(
        int precision,
        const oskar_Mem* data,
        double frequency_hz,
        double spectral_index,
        int nside,
        int galactic_coords,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
