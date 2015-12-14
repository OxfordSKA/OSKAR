/*
 * Copyright (c) 2012-2015, The University of Oxford
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


#ifndef OSKAR_GET_IMAGE_BASELINE_COORDS_H_
#define OSKAR_GET_IMAGE_BASELINE_COORDS_H_

/**
 * @file oskar_get_image_baseline_coords.h
 */

#include <oskar_global.h>

#include <oskar_mem.h>
#include "oskar_SettingsImage.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Get visibility coordinates required for imaging.
 *
 * @details
 * Populates uu and vv from visibilities structure for the particular
 * channel and time selection given the image settings of time and channel
 * range and snapshot imaging modes.
 *
 * __Note__:
 * While coordinates returned are in metres, when imaging in frequency
 * synthesis mode meters must be scaled by (lambda0/lambda) = (freq / freq0)
 * where lambda is the wavelength of the input channel and lambda0 is the
 * wavelength of the image channel.
 *
 * WARNING: This function is intended to ONLY be used with oskar_make_image().
 * WARNING: This function relies on uu, vv and ww being preallocated correctly.
 *
 * @param[out] uu            Baseline coordinates for the image, in (scaled) metres.
 * @param[out] vv            Baseline coordinates for the image, in (scaled) metres.
 * @param[out] ww            Baseline coordinates for the image, in (scaled) metres.
 * @param[in]  vis_uu        Baseline coordinates in the visibility data set, in metres.
 * @param[in]  vis_vv        Baseline coordinates in the visibility data set, in metres.
 * @param[in]  vis_ww        Baseline coordinates in the visibility data set, in metres.
 * @param[in]  num_times     Number of times in the visibility data set
 * @param[in]  num_baseline  Number of baselines in the visibility data set.
 * @param[in]  num_channels  Number of baselines in the visibility data set.
 * @param[in]  freq_start_hz Start frequency of visibility data set, in Hz.
 * @param[in]  freq_inc_hz   Frequency increment of visibility data set, in Hz.
 * @param[in]  vis_time      Time index of coordinates in the visibility data set.
 * @param[in]  im_freq       Image frequency in Hz.
 * @param[in]  settings      OSKAR image settings structure.
 *
 * @return An error code.
 */
OSKAR_EXPORT
int oskar_get_image_baseline_coords(oskar_Mem* uu, oskar_Mem* vv, oskar_Mem* ww,
        const oskar_Mem* vis_uu, const oskar_Mem* vis_vv, const oskar_Mem* vis_ww,
        int num_times, int num_baselines, int num_channels,
        double freq_start_hz, double freq_inc_hz, int vis_time, double im_freq,
        const oskar_SettingsImage* settings);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_GET_IMAGE_BASELINE_COORDS_H_ */
