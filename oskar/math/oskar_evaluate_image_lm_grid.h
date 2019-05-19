/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_EVALUATE_IMAGE_LM_GRID_H_
#define OSKAR_EVALUATE_IMAGE_LM_GRID_H_

/**
 * @file oskar_evaluate_image_lm_grid.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Generate a grid on the tangent plane suitable for use with a FITS image.
 *
 * @details
 * This function computes (orthographic) tangent-plane positions on a grid
 * that can be used to make a FITS image.
 *
 * Note that FITS images conventionally have the <b>largest</b> value of
 * RA (=longitude) and the <b>smallest</b> value of DEC (=latitude) at the
 * lowest memory address, so therefore the grid l-values must start off
 * positive and go negative, while the grid m-values start off negative
 * and go positive.
 *
 * The fastest-varying dimension is along l (longitude axis) and the slowest
 * varying is along m (latitude axis).
 *
 * The output arrays \p grid_l and \p grid_m must each be pre-sized large
 * enough to hold \p num_l * \p num_m points.
 *
 * The axes l and m are towards the East and North (in the same direction as
 * baseline dimensions u and v), respectively: see "Interferometry and
 * Synthesis in Radio Astronomy" figure 4.1 (1986 edition) or
 * figure 3.2 (2001 edition); "l and m are the direction cosines measured with
 * respect to the axes u and v." (page 71, 2001 edition).
 *
 * @param[in]  num_l   Number of required grid points in the longitude dimension.
 * @param[in]  num_m   Number of required grid points in the latitude dimension.
 * @param[in]  fov_lon The field of view in longitude (image width) in radians.
 * @param[in]  fov_lat The field of view in latitude (image height) in radians.
 * @param[out] l       The output list of l-direction cosines.
 * @param[out] m       The output list of m-direction cosines.
 */
OSKAR_EXPORT
void oskar_evaluate_image_lm_grid(int num_l, int num_m, double fov_lon,
        double fov_lat, oskar_Mem* l, oskar_Mem* m, int* status);

/**
 * @brief
 * Generate a grid on the tangent plane suitable for use with a FITS image
 * (single precision).
 *
 * @details
 * This function computes (orthographic) tangent-plane positions on a grid
 * that can be used to make a FITS image.
 *
 * Note that FITS images conventionally have the <b>largest</b> value of
 * RA (=longitude) and the <b>smallest</b> value of DEC (=latitude) at the
 * lowest memory address, so therefore the grid l-values must start off
 * positive and go negative, while the grid m-values start off negative
 * and go positive.
 *
 * The fastest-varying dimension is along l (longitude axis) and the slowest
 * varying is along m (latitude axis).
 *
 * The output arrays \p grid_l and \p grid_m must each be pre-sized large
 * enough to hold \p num_l * \p num_m points.
 *
 * The axes l and m are towards the East and North (in the same direction as
 * baseline dimensions u and v), respectively: see "Interferometry and
 * Synthesis in Radio Astronomy" figure 4.1 (1986 edition) or
 * figure 3.2 (2001 edition); "l and m are the direction cosines measured with
 * respect to the axes u and v." (page 71, 2001 edition).
 *
 * @param[in] num_l   Number of required grid points in the longitude dimension.
 * @param[in] num_m   Number of required grid points in the latitude dimension.
 * @param[in] fov_lon The field of view in longitude (image width) in radians.
 * @param[in] fov_lat The field of view in latitude (image height) in radians.
 * @param[out] grid_l The output list of l-direction cosines.
 * @param[out] grid_m The output list of m-direction cosines.
 */
OSKAR_EXPORT
void oskar_evaluate_image_lm_grid_f(int num_l, int num_m,
        float fov_lon, float fov_lat, float* grid_l, float* grid_m);

/**
 * @brief
 * Generate a grid on the tangent plane suitable for use with a FITS image
 * (double precision).
 *
 * @details
 * This function computes (orthographic) tangent-plane positions on a grid
 * that can be used to make a FITS image.
 *
 * Note that FITS images conventionally have the <b>largest</b> value of
 * RA (=longitude) and the <b>smallest</b> value of DEC (=latitude) at the
 * lowest memory address, so therefore the grid l-values must start off
 * positive and go negative, while the grid m-values start off negative
 * and go positive.
 *
 * The fastest-varying dimension is along l (longitude axis) and the slowest
 * varying is along m (latitude axis).
 *
 * The output arrays \p grid_l and \p grid_m must each be pre-sized large
 * enough to hold \p num_l * \p num_m points.
 *
 * The axes l and m are towards the East and North (in the same direction as
 * baseline dimensions u and v), respectively: see "Interferometry and
 * Synthesis in Radio Astronomy" figure 4.1 (1986 edition) or
 * figure 3.2 (2001 edition); "l and m are the direction cosines measured with
 * respect to the axes u and v." (page 71, 2001 edition).
 *
 * @param[in] num_l   Number of required grid points in the longitude dimension.
 * @param[in] num_m   Number of required grid points in the latitude dimension.
 * @param[in] fov_lon The field of view in longitude (image width) in radians.
 * @param[in] fov_lat The field of view in latitude (image height) in radians.
 * @param[out] grid_l The output list of l-direction cosines.
 * @param[out] grid_m The output list of m-direction cosines.
 */
OSKAR_EXPORT
void oskar_evaluate_image_lm_grid_d(int num_l, int num_m,
        double fov_lon, double fov_lat, double* grid_l, double* grid_m);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_IMAGE_LM_GRID_H_ */
