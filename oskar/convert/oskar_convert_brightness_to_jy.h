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

#ifndef OSKAR_CONVERT_BRIGHTNESS_TO_JY_H_
#define OSKAR_CONVERT_BRIGHTNESS_TO_JY_H_

/**
 * @file oskar_convert_brightness_to_jy.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Converts brightness values to Jy/pixel, if required.
 *
 * @details
 * Converts brightness values to Jy/pixel, if required.
 *
 * The unit strings can be either "Jy/pixel", "Jy/beam", "K" or "mK".
 *
 * @param[in,out] data             Array of pixels to convert.
 * @param[in] beam_area_pixels     Beam area, in pixels, if input is Jy/beam.
 * @param[in] pixel_area_sr        Pixel area, in steradians, if input is K.
 * @param[in] frequency_hz         Image frequency in Hz.
 * @param[in] min_peak_fraction    Minimum allowed fraction of image peak.
 * @param[in] min_abs_val          Minimum absolute value, in original units.
 * @param[in] reported_map_units   String describing units of input map.
 * @param[in] default_map_units    String describing default units of input map.
 * @param[in] override_input_units If set, override reported units with default.
 * @param[in,out] status           Status return code.
 */
OSKAR_EXPORT
void oskar_convert_brightness_to_jy(oskar_Mem* data, double beam_area_pixels,
        double pixel_area_sr, double frequency_hz, double min_peak_fraction,
        double min_abs_val, const char* reported_map_units,
        const char* default_map_units, int override_input_units, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_BRIGHTNESS_TO_JY_H_ */
