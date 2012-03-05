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

#ifndef OSKAR_CUDAK_EVALUATE_DIPOLE_PATTERN_H_
#define OSKAR_CUDAK_EVALUATE_DIPOLE_PATTERN_H_

/**
 * @file oskar_cudak_evaluate_dipole_pattern.h
 */

#include "oskar_global.h"
#include "utility/oskar_vector_types.h"

/**
 * @brief
 * Evaluates patterns of two perfect dipoles at source positions for given
 * orientations (single precision).
 *
 * @details
 * This function evaluates the patterns of two perfect dipole antennas
 * at the supplied source positions.
 *
 * The dipole axis angles should be specified as the angle East (x) from
 * North (y).
 *
 * The output matrix is
 *
 * ( g_phi^a   g_theta^a ) = ( -i Ex   0 )
 * ( g_phi^b   g_theta^b )   ( -i Ey   0 )
 *
 * where phi and theta are the angles measured from x to y and from xy to z,
 * respectively.
 *
 * @param[in] num_sources    Number of sources.
 * @param[in] l              Source direction cosines in x.
 * @param[in] m              Source direction cosines in y.
 * @param[in] n              Source direction cosines in z.
 * @param[in] orientation_x  The azimuth angle of the dipole nominally oriented along the x axis (normally PI/2).
 * @param[in] orientation_y  The azimuth angle of the dipole nominally oriented along the y axis (normally 0).
 * @param[out] pattern       Array of output Jones matrices per source.
 */
__global__
void oskar_cudak_evaluate_dipole_pattern_f(int num_sources, float* l,
        float* m, float* n, float orientation_x, float orientation_y,
        float4c* pattern);

/**
 * @brief
 * Evaluates patterns of two perfect dipoles at source positions for given
 * orientations (double precision).
 *
 * @details
 * This function evaluates the patterns of two perfect dipole antennas
 * at the supplied source positions.
 *
 * The dipole axis angles should be specified as the angle East (x) from
 * North (y).
 *
 * The output matrix is
 *
 * ( g_phi^a   g_theta^a ) = ( i Ex   0 )
 * ( g_phi^b   g_theta^b )   ( i Ey   0 )
 *
 * where phi and theta are the angles measured from x to y and from xy to z,
 * respectively.
 *
 * @param[in] num_sources    Number of sources.
 * @param[in] l              Source direction cosines in x.
 * @param[in] m              Source direction cosines in y.
 * @param[in] n              Source direction cosines in z.
 * @param[in] orientation_x  The azimuth angle of the dipole nominally oriented along the x axis (normally PI/2).
 * @param[in] orientation_y  The azimuth angle of the dipole nominally oriented along the y axis (normally 0).
 * @param[out] pattern       Array of output Jones matrices per source.
 */
__global__
void oskar_cudak_evaluate_dipole_pattern_d(int num_sources, double* l,
        double* m, double* n, double orientation_x, double orientation_y,
        double4c* pattern);

#endif /* OSKAR_CUDAK_EVALUATE_DIPOLE_PATTERN_H_ */
