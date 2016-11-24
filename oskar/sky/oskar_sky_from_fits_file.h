/*
 * Copyright (c) 2016, The University of Oxford
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

#ifndef OSKAR_SKY_FROM_FITS_FILE_H_
#define OSKAR_SKY_FROM_FITS_FILE_H_

/**
 * @file oskar_sky_from_fits_file.h
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
oskar_Sky* oskar_sky_from_fits_file(int precision, const char* filename,
        double min_peak_fraction, double min_abs_val,
        const char* default_map_units, int override_units, double frequency_hz,
        double spectral_index, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_FROM_FITS_FILE_H_ */
