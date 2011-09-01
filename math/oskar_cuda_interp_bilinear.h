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

#ifndef OSKAR_MATH_CUDA_INTERP_BILINEAR_H_
#define OSKAR_MATH_CUDA_INTERP_BILINEAR_H_

/**
 * @file oskar_math_cuda_interp_bilinear.h
 */

#include "oskar_windows.h"
#include "utility/oskar_vector_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Performs bilinear interpolation of real data.
 *
 * @details
 * This function performs bilinear interpolation using the native
 * texture interpolation capabilities of the graphics hardware.
 *
 * Note that the input data must have been previously allocated using
 * cudaMallocPitch().
 *
 * The positions are specified in the input \p pos array, which contains
 * (x, y) coordinate pairs in units of (width, height) respectively.
 * The range is (0, width - 1), (0, height - 1), and the centre of the first
 * element is at (0.5, 0.5).
 *
 * @param[in] width   Width (fastest varying dimension) of lookup table.
 * @param[in] height  Height (slowest varying dimension) of lookup table.
 * @param[in] pitch   Memory pitch of input, as returned by cudaMallocPitch().
 * @param[in] input   Input lookup table, as returned by cudaMallocPitch().
 * @param[in] n       Number of output (interpolated) points.
 * @param[in] pos     Positions of output points.
 * @param[out] output Interpolated output data.
 */
DllExport
int oskar_cuda_interp_bilinear_f(int width, int height, int pitch,
        const float* input, int n, const float* pos_x, const float* pos_y,
        float* output);

/**
 * @brief
 * Performs bilinear interpolation of complex data.
 *
 * @details
 * This function performs bilinear interpolation of complex data using the
 * native texture interpolation capabilities of the graphics hardware.
 *
 * Note that the input data must have been previously allocated using
 * cudaMallocPitch().
 *
 * The positions are specified in the input \p pos array, which contains
 * (x, y) coordinate pairs in units of (width, height) respectively.
 * The range is (0, width - 1), (0, height - 1), and the centre of the first
 * element is at (0.5, 0.5).
 *
 * @param[in] width   Width (fastest varying dimension) of lookup table.
 * @param[in] height  Height (slowest varying dimension) of lookup table.
 * @param[in] pitch   Memory pitch of input, as returned by cudaMallocPitch().
 * @param[in] input   Input lookup table, as returned by cudaMallocPitch().
 * @param[in] n       Number of output (interpolated) points.
 * @param[in] pos     Positions of output points.
 * @param[out] output Interpolated output data.
 */
DllExport
int oskar_cuda_interp_bilinear_complex_f(int width, int height, int pitch,
        const float2* input, int n, const float* pos_x, const float* pos_y,
        float2* output);


#ifdef __cplusplus
}
#endif

#endif // OSKAR_MATH_CUDA_INTERP_BILINEAR_H_
