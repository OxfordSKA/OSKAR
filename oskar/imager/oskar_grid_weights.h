/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
        const double* RESTRICT uu, const double* RESTRICT vv,
        const double* RESTRICT weight, const double cell_size_rad,
        const int grid_size, size_t* RESTRICT num_skipped,
        double* RESTRICT grid);

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
 * @param[in] grid              Gridded weights.
 */
OSKAR_EXPORT
void oskar_grid_weights_read_d(const size_t num_points,
        const double* RESTRICT uu, const double* RESTRICT vv,
        const double* RESTRICT weight_in, double* RESTRICT weight_out,
        const double cell_size_rad, const int grid_size,
        size_t* RESTRICT num_skipped, const double* RESTRICT grid);

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
 * @param[in,out] grid_guard    Guard digits for gridded weights.
 */
OSKAR_EXPORT
void oskar_grid_weights_write_f(const size_t num_points,
        const float* RESTRICT uu, const float* RESTRICT vv,
        const float* RESTRICT weight, const float cell_size_rad,
        const int grid_size, size_t* RESTRICT num_skipped,
        float* RESTRICT grid, float* RESTRICT grid_guard);

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
 * @param[in] grid              Gridded weights.
 */
OSKAR_EXPORT
void oskar_grid_weights_read_f(const size_t num_points,
        const float* RESTRICT uu, const float* RESTRICT vv,
        const float* RESTRICT weight_in, float* RESTRICT weight_out,
        const float cell_size_rad, const int grid_size,
        size_t* RESTRICT num_skipped, const float* RESTRICT grid);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
