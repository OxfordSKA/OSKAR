/*
 * Copyright (c) 2019, The University of Oxford
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

#ifndef OSKAR_EVALUATE_TEC_SCREEN_H_
#define OSKAR_EVALUATE_TEC_SCREEN_H_

/**
 * @file oskar_evaluate_tec_screen.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Evaluates a TEC screen at the given source positions.
 *
 * @details
 * Evaluates a TEC screen at the given source positions.
 *
 * @param[in] num_points      Number of points at which to evaluate the screen.
 * @param[in] l               Source l-direction cosines.
 * @param[in] m               Source m-direction cosines.
 * @param[in] station_u_m     Station u-coordinate, in metres.
 * @param[in] station_v_m     Station v-coordinate, in metres.
 * @param[in] frequency_hz    Frequency, in Hz.
 * @param[in] screen_height_m Height of phase screen above array, in metres.
 * @param[in] screen_pixel_size_m Size of each pixel, in metres.
 * @param[in] screen_num_pixels_x Number of pixels along the x-dimension.
 * @param[in] screen_num_pixels_x Number of pixels along the y-dimension.
 * @param[in] tec_screen      TEC screen to evaluate.
 * @param[in] offset_out      Start offset into output array.
 * @param[out] out            Complex output array.
 * @param[in,out] status      Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_tec_screen(
        int num_points,
        const oskar_Mem* l,
        const oskar_Mem* m,
        double station_u_m,
        double station_v_m,
        double frequency_hz,
        double screen_height_m,
        double screen_pixel_size_m,
        int screen_num_pixels_x,
        int screen_num_pixels_y,
        const oskar_Mem* tec_screen,
        int offset_out,
        oskar_Mem* out,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
