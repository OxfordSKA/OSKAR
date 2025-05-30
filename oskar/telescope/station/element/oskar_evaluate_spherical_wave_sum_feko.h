/*
 * Copyright (c) 2023, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_EVALUATE_SPHERICAL_WAVE_SUM_FEKO_H_
#define OSKAR_EVALUATE_SPHERICAL_WAVE_SUM_FEKO_H_

/**
 * @file oskar_evaluate_spherical_wave_sum_feko.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluate the sum of spherical wave coefficients.
 *
 * @details
 * Evaluate the sum of spherical wave coefficients at the
 * given coordinates.
 *
 * @param[in] num_points    Number of coordinate points.
 * @param[in] theta         Coordinate theta (polar) values, in radians.
 * @param[in] phi_x         Coordinate phi (azimuthal) values for X, in radians.
 * @param[in] phi_y         Coordinate phi (azimuthal) values for Y, in radians.
 * @param[in] l_max         Maximum order of spherical wave.
 * @param[in] alpha         TE and TM mode coefficients for X and Y antennas.
 * @param[in] swap_xy       If true, swap the X and Y antenna in the output.
 * @param[in] offset_out    Offset into output data array.
 * @param[in,out] pattern   Output data array of length at least \p num_points.
 * @param[in,out] status    Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_spherical_wave_sum_feko(
        int num_points,
        const oskar_Mem* theta,
        const oskar_Mem* phi_x,
        const oskar_Mem* phi_y,
        int l_max,
        const oskar_Mem* alpha,
        int swap_xy,
        int offset_out,
        oskar_Mem* pattern,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
