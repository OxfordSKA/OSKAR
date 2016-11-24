/*
 * Copyright (c) 2013-2016, The University of Oxford
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

#ifndef OSKAR_EVALUATE_IMAGE_LMN_GRID_H_
#define OSKAR_EVALUATE_IMAGE_LMN_GRID_H_

/**
 * @file oskar_evaluate_image_lmn_grid.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Generate a grid of evenly spaced direction cosines on the tangent plane.
 *
 * @details
 * Note that n directions are returned as n-1 (distance from the tangent plane).
 *
 * @param[in] num_l   Number of required grid points in the longitude dimension.
 * @param[in] num_m   Number of required grid points in the latitude dimension.
 * @param[in] fov_lon The field of view in longitude (image width) in radians.
 * @param[in] fov_lat The field of view in latitude (image height) in radians.
 * @param[in] centred If true, the grid will be symmetric about zero.
 * @param[out] grid_l The output list of l-direction cosines.
 * @param[out] grid_m The output list of m-direction cosines.
 * @param[out] grid_n The output list of n-direction cosines.
 */
OSKAR_EXPORT
void oskar_evaluate_image_lmn_grid(int num_l, int num_m, double fov_lon,
        double fov_lat, int centred, oskar_Mem* grid_l, oskar_Mem* grid_m,
        oskar_Mem* grid_n, int* status);

/**
 * @brief
 * Generate a grid of evenly spaced direction cosines on the tangent plane
 * (single precision).
 *
 * @details
 * Note that n directions are returned as n-1 (distance from the tangent plane).
 *
 * @param[in] num_l   Number of required grid points in the longitude dimension.
 * @param[in] num_m   Number of required grid points in the latitude dimension.
 * @param[in] fov_lon The field of view in longitude (image width) in radians.
 * @param[in] fov_lat The field of view in latitude (image height) in radians.
 * @param[in] centred If true, the grid will be symmetric about zero.
 * @param[out] grid_l The output list of l-direction cosines.
 * @param[out] grid_m The output list of m-direction cosines.
 * @param[out] grid_n The output list of n-direction cosines.
 */
OSKAR_EXPORT
void oskar_evaluate_image_lmn_grid_f(int num_l, int num_m, float fov_lon,
        float fov_lat, int centred, float* grid_l, float* grid_m,
        float* grid_n);

/**
 * @brief
 * Generate a grid of evenly spaced direction cosines on the tangent plane
 * (double precision).
 *
 * @details
 * Note that n directions are returned as n-1 (distance from the tangent plane).
 *
 * @param[in] num_l   Number of required grid points in the longitude dimension.
 * @param[in] num_m   Number of required grid points in the latitude dimension.
 * @param[in] fov_lon The field of view in longitude (image width) in radians.
 * @param[in] fov_lat The field of view in latitude (image height) in radians.
 * @param[in] centred If true, the grid will be symmetric about zero.
 * @param[out] grid_l The output list of l-direction cosines.
 * @param[out] grid_m The output list of m-direction cosines.
 * @param[out] grid_n The output list of n-direction cosines.
 */
OSKAR_EXPORT
void oskar_evaluate_image_lmn_grid_d(int num_l, int num_m, double fov_lon,
        double fov_lat, int centred, double* grid_l, double* grid_m,
        double* grid_n);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_IMAGE_LMN_GRID_H_ */
