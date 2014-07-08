/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#ifndef OSKAR_EVALUATE_GEOMETRIC_DIPOLE_PATTERN_H_
#define OSKAR_EVALUATE_GEOMETRIC_DIPOLE_PATTERN_H_

/**
 * @file oskar_evaluate_geometric_dipole_pattern.h
 */

#include <oskar_global.h>
#include <oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates pattern of a perfect dipole at source positions
 * (single precision).
 *
 * @details
 * This function evaluates the pattern of a perfect dipole antenna
 * at the supplied source positions.
 *
 * The supplied theta and phi positions of the sources are the <b>modified</b>
 * source positions. They must be adjusted relative to a dipole with its axis
 * oriented along the x-direction.
 *
 * @param[in] num_points         Number of points.
 * @param[in] theta              Point position (modified) theta values in rad.
 * @param[in] phi                Point position (modified) phi values in rad.
 * @param[in] stride             Stride into output arrays (normally 4).
 * @param[out] E_theta           Response per point in E_theta.
 * @param[out] E_phi             Response per point in E_phi.
 */
OSKAR_EXPORT
void oskar_evaluate_geometric_dipole_pattern_f(int num_points,
        const float* theta, const float* phi, int stride,
        float2* E_theta, float2* E_phi);

/**
 * @brief
 * Evaluates pattern of a perfect dipole at source positions
 * (scalar version, single precision).
 *
 * @details
 * This function evaluates the scalar pattern of a perfect dipole antenna
 * at the supplied source positions.
 *
 * The supplied theta and phi positions of the sources are the <b>modified</b>
 * source positions. They must be adjusted relative to a dipole with its axis
 * oriented along the x-direction.
 *
 * @param[in] num_points         Number of points.
 * @param[in] theta              Point position (modified) theta values in rad.
 * @param[in] phi                Point position (modified) phi values in rad.
 * @param[in] stride             Stride into output array (normally 1).
 * @param[out] pattern           Response per point.
 */
OSKAR_EXPORT
void oskar_evaluate_geometric_dipole_pattern_scalar_f(int num_points,
        const float* theta, const float* phi, int stride, float2* pattern);

/**
 * @brief
 * Evaluates pattern of a perfect dipole at source positions
 * (double precision).
 *
 * @details
 * This function evaluates the pattern of a perfect dipole antenna
 * at the supplied source positions.
 *
 * The supplied theta and phi positions of the sources are the <b>modified</b>
 * source positions. They must be adjusted relative to a dipole with its axis
 * oriented along the x-direction.
 *
 * @param[in] num_points         Number of points.
 * @param[in] theta              Point position (modified) theta values in rad.
 * @param[in] phi                Point position (modified) phi values in rad.
 * @param[in] stride             Stride into output arrays (normally 4).
 * @param[out] E_theta           Response per point in E_theta.
 * @param[out] E_phi             Response per point in E_phi.
 */
OSKAR_EXPORT
void oskar_evaluate_geometric_dipole_pattern_d(int num_points,
        const double* theta, const double* phi, int stride,
        double2* E_theta, double2* E_phi);

/**
 * @brief
 * Evaluates pattern of a perfect dipole at source positions
 * (scalar version, double precision).
 *
 * @details
 * This function evaluates the scalar pattern of a perfect dipole antenna
 * at the supplied source positions.
 *
 * The supplied theta and phi positions of the sources are the <b>modified</b>
 * source positions. They must be adjusted relative to a dipole with its axis
 * oriented along the x-direction.
 *
 * @param[in] num_points         Number of points.
 * @param[in] theta              Point position (modified) theta values in rad.
 * @param[in] phi                Point position (modified) phi values in rad.
 * @param[in] stride             Stride into output array (normally 1).
 * @param[out] pattern           Response per point.
 */
OSKAR_EXPORT
void oskar_evaluate_geometric_dipole_pattern_scalar_d(int num_points,
        const double* theta, const double* phi, int stride, double2* pattern);

/**
 * @brief
 * Evaluates pattern of a perfect dipole at source positions.
 *
 * @details
 * This function evaluates the pattern of a perfect dipole antenna
 * at the supplied source positions.
 *
 * The supplied theta and phi positions of the sources are the <b>modified</b>
 * source positions. They must be adjusted relative to a dipole with its axis
 * oriented along the x-direction.
 *
 * @param[out] pattern           Array of output Jones matrices/scalars per source.
 * @param[in] theta              Point position (modified) theta values in rad.
 * @param[in] phi                Point position (modified) phi values in rad.
 * @param[in] offset             Offset index into output arrays.
 * @param[in] stride             Stride into output array (normally 1 or 4).
 * @param[in,out] status         Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_geometric_dipole_pattern(oskar_Mem* pattern, int num_points,
        const oskar_Mem* theta, const oskar_Mem* phi, int offset, int stride,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_GEOMETRIC_DIPOLE_PATTERN_H_ */
