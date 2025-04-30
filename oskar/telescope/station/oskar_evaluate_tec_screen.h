/*
 * Copyright (c) 2019-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
 * If the output array is a matrix type, then the effect of Faraday
 * rotation will be included.
 * Otherwise, the output is a scalar phase term only.
 *
 * @param[in] isoplanatic     If true, treat screen as isoplanatic.
 * @param[in] num_points      Number of points at which to evaluate the screen.
 * @param[in] l               Source l-direction cosines.
 * @param[in] m               Source m-direction cosines.
 * @param[in] hor_x           Source direction cosines along local x (East).
 * @param[in] hor_y           Source direction cosines along local y (North).
 * @param[in] hor_z           Source direction cosines along local z (Up).
 * @param[in] station_u_m     Station u-coordinate, in metres.
 * @param[in] station_v_m     Station v-coordinate, in metres.
 * @param[in] frequency_hz    Frequency, in Hz.
 * @param[in] field_x         Component of magnetic field along x, in nT.
 * @param[in] field_y         Component of magnetic field along y, in nT.
 * @param[in] field_z         Component of magnetic field along z, in nT.
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
        int isoplanatic,
        int num_points,
        const oskar_Mem* l,
        const oskar_Mem* m,
        const oskar_Mem* hor_x,
        const oskar_Mem* hor_y,
        const oskar_Mem* hor_z,
        double station_u_m,
        double station_v_m,
        double frequency_hz,
        double field_x,
        double field_y,
        double field_z,
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
