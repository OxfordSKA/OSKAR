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

#ifndef OSKAR_CUDA_INTERP_BILINEAR_H_
#define OSKAR_CUDA_INTERP_BILINEAR_H_

/**
 * @file oskar_cuda_interp_bilinear.h
 */

#include "oskar_global.h"
#include "utility/oskar_vector_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Performs bilinear interpolation (single precision real).
 *
 * @details
 * This function performs bilinear interpolation using the native
 * texture interpolation capabilities of the graphics hardware.
 *
 * Note that the input data must have been allocated using cudaMallocPitch().
 *
 * The positions are specified in the input \p pos_x and \p pos_y arrays, which
 * contain normalised (x, y) coordinates. The coordinates must both be in the
 * range [0.0, 1.0), and correspond to the full range of the input texture
 * dimensions. <b>Note that any coordinate values outside this range will be
 * clamped.</b>
 *
 * @param[in] size_x    Width (fastest varying dimension) of lookup table.
 * @param[in] size_y    Height (slowest varying dimension) of lookup table.
 * @param[in] pitch     Memory pitch of input, as returned by cudaMallocPitch().
 * @param[in] d_input   Lookup table pointer, as returned by cudaMallocPitch().
 * @param[in] n         Number of output (interpolated) points.
 * @param[in] d_pos_x   The x-positions of the output points.
 * @param[in] d_pos_y   The y-positions of the output points.
 * @param[out] d_output Interpolated output data.
 */
OSKAR_EXPORT
int oskar_cuda_interp_bilinear_f(int size_x, int size_y, int pitch,
        const float* d_input, int n, const float* d_pos_x,
        const float* d_pos_y, float* d_output);

/**
 * @brief
 * Performs bilinear interpolation (single precision complex).
 *
 * @details
 * This function performs bilinear interpolation using the native
 * texture interpolation capabilities of the graphics hardware.
 *
 * Note that the input data must have been allocated using cudaMallocPitch().
 *
 * The positions are specified in the input \p pos_x and \p pos_y arrays, which
 * contain normalised (x, y) coordinates. The coordinates must both be in the
 * range [0.0, 1.0), and correspond to the full range of the input texture
 * dimensions. <b>Note that any coordinate values outside this range will be
 * clamped.</b>
 *
 * @param[in] size_x    Width (fastest varying dimension) of lookup table.
 * @param[in] size_y    Height (slowest varying dimension) of lookup table.
 * @param[in] pitch     Memory pitch of input, as returned by cudaMallocPitch().
 * @param[in] d_input   Lookup table pointer, as returned by cudaMallocPitch().
 * @param[in] n         Number of output (interpolated) points.
 * @param[in] d_pos_x   The x-positions of the output points.
 * @param[in] d_pos_y   The y-positions of the output points.
 * @param[out] d_output Interpolated output data.
 */
OSKAR_EXPORT
int oskar_cuda_interp_bilinear_c(int size_x, int size_y, int pitch,
        const float2* d_input, int n, const float* d_pos_x,
        const float* d_pos_y, float2* d_output);

/**
 * @brief
 * Performs bilinear interpolation (double precision real).
 *
 * @details
 * This function performs bilinear interpolation using the native
 * texture interpolation capabilities of the graphics hardware. <b>Although the
 * data returned is in double precision, current architecture limitations mean
 * that the interpolation itself is still carried out in single precision.</b>
 *
 * Note that the input data must have been allocated using cudaMallocPitch().
 *
 * The positions are specified in the input \p pos_x and \p pos_y arrays, which
 * contain normalised (x, y) coordinates. The coordinates must both be in the
 * range [0.0, 1.0), and correspond to the full range of the input texture
 * dimensions. <b>Note that any coordinate values outside this range will be
 * clamped.</b>
 *
 * @param[in] size_x    Width (fastest varying dimension) of lookup table.
 * @param[in] size_y    Height (slowest varying dimension) of lookup table.
 * @param[in] pitch     Memory pitch of input, as returned by cudaMallocPitch().
 * @param[in] d_input   Lookup table pointer, as returned by cudaMallocPitch().
 * @param[in] n         Number of output (interpolated) points.
 * @param[in] d_pos_x   The x-positions of the output points.
 * @param[in] d_pos_y   The y-positions of the output points.
 * @param[out] d_output Interpolated output data.
 */
OSKAR_EXPORT
int oskar_cuda_interp_bilinear_d(int size_x, int size_y, int pitch,
        const float* d_input, int n, const double* d_pos_x,
        const double* d_pos_y, double* d_output);

/**
 * @brief
 * Performs bilinear interpolation (double precision complex).
 *
 * @details
 * This function performs bilinear interpolation using the native
 * texture interpolation capabilities of the graphics hardware. <b>Although the
 * data returned is in double precision, current architecture limitations mean
 * that the interpolation itself is still carried out in single precision.</b>
 *
 * Note that the input data must have been allocated using cudaMallocPitch().
 *
 * The positions are specified in the input \p pos_x and \p pos_y arrays, which
 * contain normalised (x, y) coordinates. The coordinates must both be in the
 * range [0.0, 1.0), and correspond to the full range of the input texture
 * dimensions. <b>Note that any coordinate values outside this range will be
 * clamped.</b>
 *
 * @param[in] size_x    Width (fastest varying dimension) of lookup table.
 * @param[in] size_y    Height (slowest varying dimension) of lookup table.
 * @param[in] pitch     Memory pitch of input, as returned by cudaMallocPitch().
 * @param[in] d_input   Lookup table pointer, as returned by cudaMallocPitch().
 * @param[in] n         Number of output (interpolated) points.
 * @param[in] d_pos_x   The x-positions of the output points.
 * @param[in] d_pos_y   The y-positions of the output points.
 * @param[out] d_output Interpolated output data.
 */
OSKAR_EXPORT
int oskar_cuda_interp_bilinear_z(int size_x, int size_y, int pitch,
        const float2* d_input, int n, const double* d_pos_x,
        const double* d_pos_y, double2* d_output);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CUDA_INTERP_BILINEAR_H_ */
