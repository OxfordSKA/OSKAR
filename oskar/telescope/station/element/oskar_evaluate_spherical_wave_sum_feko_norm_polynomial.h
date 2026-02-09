/*
 * Copyright (c) 2025-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_EVALUATE_SPHERICAL_WAVE_SUM_FEKO_NORM_POLYNOMIAL_H_
#define OSKAR_EVALUATE_SPHERICAL_WAVE_SUM_FEKO_NORM_POLYNOMIAL_H_

/**
 * @file oskar_evaluate_spherical_wave_sum_feko_norm_polynomial.h
 */

#include "oskar_global.h"
#include "mem/oskar_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluate the sum of FEKO spherical wave coefficients using
 * normalised Legendre polynomials.
 *
 * @details
 * Evaluate the sum of FEKO spherical wave coefficients at the
 * given coordinates.
 * This version uses normalised Legendre polynomials to allow use of
 * high-order spherical waves.
 *
 * @param[in] num_points Number of coordinate points.
 * @param[in] theta_rad Coordinate theta (polar) values, in radians.
 * @param[in] phi_x_rad Coordinate phi (azimuthal) values for X, in radians.
 * @param[in] phi_y_rad Coordinate phi (azimuthal) values for Y, in radians.
 * @param[in] n_max Maximum order of spherical wave.
 * @param root_n Pre-computed square roots, length 2 * \p n_max + 1.
 * @param use_ticra_convention If true, conjugate and scale coefficients
 *                             by sqrt(8 * pi), and index m-values according
 *                             to TICRA convention.
 * @param[in] coeffs TE and TM mode coefficients for X and Y antennas.
 * @param workspace Pre-allocated array to use as workspace,
 *                  size ( \p n_max + 2 ) * \p num_points.
 * @param[in] swap_xy If true, swap the X and Y antenna in the output.
 * @param[in] offset_out Offset into output data array.
 * @param[in,out] pattern Output data array of length at least \p num_points.
 * @param[in,out] status Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_spherical_wave_sum_feko_norm_polynomial(
        int num_points,
        const oskar_Mem* theta_rad,
        const oskar_Mem* phi_x_rad,
        const oskar_Mem* phi_y_rad,
        int n_max,
        const oskar_Mem* root_n,
        int use_ticra_convention,
        const oskar_Mem* coeffs,
        oskar_Mem* workspace,
        int swap_xy,
        int offset_out,
        oskar_Mem* pattern,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
