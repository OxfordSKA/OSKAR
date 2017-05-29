/*
 * Copyright (c) 2016-2017, The University of Oxford
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

#ifndef OSKAR_GRID_WEIGHTS_H_
#define OSKAR_GRID_WEIGHTS_H_

/**
 * @file oskar_grid_weights.h
 */

#include <oskar_global.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Updates gridded weights (double precision).
 *
 * @details
 * Updates gridded weights for the supplied visibility points.
 *
 * @param[in] num_points        Number of data points.
 * @param[in] uu                Baseline uu coordinates, in wavelengths.
 * @param[in] vv                Baseline vv coordinates, in wavelengths.
 * @param[in] weight            Input visibility weights.
 * @param[in] cell_size_rad     Cell size, in radians.
 * @param[in] grid_size         Side length of grid.
 * @param[out] num_skipped      Number of points that fell outside the grid.
 * @param[in,out] grid          Gridded weights.
 */
OSKAR_EXPORT
void oskar_grid_weights_write_d(const size_t num_points,
        const double* restrict uu, const double* restrict vv,
        const double* restrict weight, const double cell_size_rad,
        const int grid_size, size_t* restrict num_skipped,
        double* restrict grid);

/**
 * @brief
 * Re-weights visibilities using gridded weights (double precision).
 *
 * @details
 * Re-weights supplied visibilities using gridded weights, for
 * uniform weighting.
 *
 * @param[in] num_points        Number of data points.
 * @param[in] uu                Baseline uu coordinates, in wavelengths.
 * @param[in] vv                Baseline vv coordinates, in wavelengths.
 * @param[in] weight            Input visibility weights.
 * @param[in] cell_size_rad     Cell size, in radians.
 * @param[in] grid_size         Side length of grid.
 * @param[out] num_skipped      Number of points that fell outside the grid.
 * @param[in,out] grid          Gridded weights.
 */
OSKAR_EXPORT
void oskar_grid_weights_read_d(const size_t num_points,
        const double* restrict uu, const double* restrict vv,
        const double* restrict weight_in, double* restrict weight_out,
        const double cell_size_rad, const int grid_size,
        size_t* restrict num_skipped, const double* restrict grid);

/**
 * @brief
 * Updates gridded weights (single precision).
 *
 * @details
 * Updates gridded weights for the supplied visibility points.
 *
 * @param[in] num_points        Number of data points.
 * @param[in] uu                Baseline uu coordinates, in wavelengths.
 * @param[in] vv                Baseline vv coordinates, in wavelengths.
 * @param[in] weight            Input visibility weights.
 * @param[in] cell_size_rad     Cell size, in radians.
 * @param[in] grid_size         Side length of grid.
 * @param[out] num_skipped      Number of points that fell outside the grid.
 * @param[in,out] grid          Gridded weights.
 */
OSKAR_EXPORT
void oskar_grid_weights_write_f(const size_t num_points,
        const float* restrict uu, const float* restrict vv,
        const float* restrict weight, const float cell_size_rad,
        const int grid_size, size_t* restrict num_skipped,
        float* restrict grid);

/**
 * @brief
 * Re-weights visibilities using gridded weights (single precision).
 *
 * @details
 * Re-weights supplied visibilities using gridded weights, for
 * uniform weighting.
 *
 * @param[in] num_points        Number of data points.
 * @param[in] uu                Baseline uu coordinates, in wavelengths.
 * @param[in] vv                Baseline vv coordinates, in wavelengths.
 * @param[in] weight            Input visibility weights.
 * @param[in] cell_size_rad     Cell size, in radians.
 * @param[in] grid_size         Side length of grid.
 * @param[out] num_skipped      Number of points that fell outside the grid.
 * @param[in,out] grid          Gridded weights.
 */
OSKAR_EXPORT
void oskar_grid_weights_read_f(const size_t num_points,
        const float* restrict uu, const float* restrict vv,
        const float* restrict weight_in, float* restrict weight_out,
        const float cell_size_rad, const int grid_size,
        size_t* restrict num_skipped, const float* restrict grid);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_GRID_WEIGHTS_H_ */
