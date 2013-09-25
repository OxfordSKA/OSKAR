/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#ifndef OSKAR_EVALUATE_ARRAY_PATTERN_DIPOLES_CUDA_H_
#define OSKAR_EVALUATE_ARRAY_PATTERN_DIPOLES_CUDA_H_

/**
 * @file oskar_evaluate_array_pattern_dipoles_cuda.h
 */

#include <oskar_global.h>
#include <oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_EXPORT
void oskar_evaluate_array_pattern_dipoles_cuda_f(int num_antennas,
        float wavenumber, const float* x, const float* y, const float* z,
        const float* cos_orientation_x, const float* sin_orientation_x,
        const float* cos_orientation_y, const float* sin_orientation_y,
        const float2* weights, int num_sources, const float* l,
        const float* m, const float* n, float4c* pattern);

OSKAR_EXPORT
void oskar_evaluate_array_pattern_dipoles_cuda_d(int num_antennas,
        double wavenumber, const double* x, const double* y, const double* z,
        const double* cos_orientation_x, const double* sin_orientation_x,
        const double* cos_orientation_y, const double* sin_orientation_y,
        const double2* weights, int num_sources, const double* l,
        const double* m, const double* n, double4c* pattern);

#ifdef __CUDACC__

/**
 * @brief
 * Evaluates an aperture array station beam, assuming the array is composed of
 * perfect dipoles (single precision).
 *
 * @details
 * This function evaluates a beam from an aperture array of perfect dipoles
 * at the supplied source positions.
 *
 * The dipole orientation angles specify the dipole axis as the angle
 * East (x) from North (y).
 *
 * The output matrix is
 *
 * ( e_theta^a   e_phi^a )
 * ( e_theta^b   e_phi^b )
 *
 * where phi and theta are the angles measured from x to y and from xy to z,
 * respectively.
 *
 * The 'a' dipole is nominally along the x axis, and
 * the 'b' dipole is nominally along the y axis.
 * The azimuth orientation of 'a' should normally be 90 degrees, and
 * the azimuth orientation of 'b' should normally be 0 degrees.
 *
 * The station beam is evaluated using a DFT.
 *
 * @param[in] num_sources        Number of sources.
 * @param[in] wavenumber         Wavenumber (2 pi / wavelength).
 * @param[in] l                  Source direction cosines in x.
 * @param[in] m                  Source direction cosines in y.
 * @param[in] n                  Source direction cosines in z.
 * @param[in] cos_orientation_x  The cosine of the azimuth angle of nominal x dipole.
 * @param[in] sin_orientation_x  The sine of the azimuth angle of nominal x dipole.
 * @param[in] cos_orientation_y  The cosine of the azimuth angle of nominal y dipole.
 * @param[in] sin_orientation_y  The sine of the azimuth angle of nominal y dipole.
 * @param[out] pattern           Array of output Jones matrices per source.
 */
__global__
void oskar_evaluate_array_pattern_dipoles_cudak_f(const int num_antennas,
        const float wavenumber, const float* x, const float* y, const float* z,
        const float* cos_orientation_x, const float* sin_orientation_x,
        const float* cos_orientation_y, const float* sin_orientation_y,
        const float2* weights, const int num_sources, const float* l,
        const float* m, const float* n, const int max_in_chunk,
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
 * The dipole orientation angles specify the dipole axis as the angle
 * East (x) from North (y).
 *
 * The output matrix is
 *
 * ( e_theta^a   e_phi^a )
 * ( e_theta^b   e_phi^b )
 *
 * where phi and theta are the angles measured from x to y and from xy to z,
 * respectively.
 *
 * The 'a' dipole is nominally along the x axis, and
 * the 'b' dipole is nominally along the y axis.
 * The azimuth orientation of 'a' should normally be 90 degrees, and
 * the azimuth orientation of 'b' should normally be 0 degrees.
 *
 * @param[in] num_sources        Number of sources.
 * @param[in] wavenumber         Wavenumber (2 pi / wavelength).
 * @param[in] l                  Source direction cosines in x.
 * @param[in] m                  Source direction cosines in y.
 * @param[in] n                  Source direction cosines in z.
 * @param[in] cos_orientation_x  The cosine of the azimuth angle of nominal x dipole.
 * @param[in] sin_orientation_x  The sine of the azimuth angle of nominal x dipole.
 * @param[in] cos_orientation_y  The cosine of the azimuth angle of nominal y dipole.
 * @param[in] sin_orientation_y  The sine of the azimuth angle of nominal y dipole.
 * @param[out] pattern           Array of output Jones matrices per source.
 */
__global__
void oskar_evaluate_array_pattern_dipoles_cudak_d(const int num_antennas,
        const double wavenumber, const double* x, const double* y,
        const double* z, const double* cos_orientation_x,
        const double* sin_orientation_x, const double* cos_orientation_y,
        const double* sin_orientation_y, const double2* weights,
        const int num_sources, const double* l, const double* m,
        const double* n, const int max_in_chunk, double4c* pattern);

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_ARRAY_PATTERN_DIPOLES_CUDA_H_ */
