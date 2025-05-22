/*
 * Copyright (c) 2012-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_ELEMENT_EVALUATE_H_
#define OSKAR_ELEMENT_EVALUATE_H_

/**
 * @file oskar_element_evaluate.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates the element model at the given source positions.
 *
 * @details
 * This function evaluates the element pattern model at the given source
 * positions.
 *
 * @param[in] model         Pointer to element model structure.
 * @param[in] normalise     If true, normalise pattern to value at zenith.
 * @param[in] swap_xy       If true, swap X and Y responses in output.
 * @param[in] orientation_x Azimuth of X dipole in radians.
 * @param[in] orientation_y Azimuth of Y dipole in radians.
 * @param[in] offset_points Start offset into input coordinate arrays.
 * @param[in] num_points    Number of points at which to evaluate beam.
 * @param[in] x             Pointer to x-direction cosines.
 * @param[in] y             Pointer to y-direction cosines.
 * @param[in] z             Pointer to z-direction cosines.
 * @param[in] frequency_hz  Current observing frequency in Hz.
 * @param[in,out] theta     Pointer to work array for computing theta values.
 * @param[in,out] phi_x     Pointer to work array for computing phi values.
 * @param[in,out] phi_y     Pointer to work array for computing phi values.
 * @param[in] offset_out    Start offset into output array.
 * @param[in,out] output    Pointer to output array.
 * @param[in,out] status    Status return code.
 */
OSKAR_EXPORT
void oskar_element_evaluate(
        const oskar_Element* model,
        int normalise,
        int swap_xy,
        double orientation_x,
        double orientation_y,
        int offset_points,
        int num_points,
        const oskar_Mem* x,
        const oskar_Mem* y,
        const oskar_Mem* z,
        double frequency_hz,
        oskar_Mem* theta,
        oskar_Mem* phi_x,
        oskar_Mem* phi_y,
        int offset_out,
        oskar_Mem* output,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
