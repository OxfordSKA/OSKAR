/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
 *                                 A value of zero or less disables this filter.
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

#endif /* include guard */
