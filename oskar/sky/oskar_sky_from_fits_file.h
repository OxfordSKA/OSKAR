/*
 * Copyright (c) 2016-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_FROM_FITS_FILE_H_
#define OSKAR_SKY_FROM_FITS_FILE_H_

/**
 * @file oskar_sky_from_fits_file.h
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
 * The \p default_map_units can be either "Jy/beam", "Jy/pixel", "K" or "mK".
 *
 * @param[in] precision         Enumerated precision of the output sky model.
 * @param[in] filename          Pathname of FITS file to load.
 * @param[in] min_peak_fraction Minimum allowed fraction of image peak.
 * @param[in] min_abs_val       Ignore pixels below this value.
 * @param[in] default_map_units Map units, if not found from the file.
 * @param[in] override_units    If set, override map units with the default.
 * @param[in] frequency_hz      Frequency of image data, in Hz, if not found.
 * @param[in] spectral_index    Spectral index to give each pixel.
 * @param[in,out] status        Status return code.
 */
OSKAR_EXPORT
oskar_Sky* oskar_sky_from_fits_file(
        int precision,
        const char* filename,
        double min_peak_fraction,
        double min_abs_val,
        const char* default_map_units,
        int override_units,
        double frequency_hz,
        double spectral_index,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
