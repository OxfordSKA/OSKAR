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

#ifndef OSKAR_BF_CUDAK_ANTENNA_H_
#define OSKAR_BF_CUDAK_ANTENNA_H_

/**
 * @file oskar_bf_cudak_antenna.h
 */

#include "oskar_global.h"

/**
 * @brief
 * CUDA kernel to apply Gaussian antenna response to beam pattern
 * (single precision).
 *
 * @details
 * This kernel multiplies the complex beam pattern by a Gaussian of a size
 * corresponding to the given parameters.
 *
 * @param[in] ns Number of pixels (source positions).
 * @param[in] se Source elevation positions in radians.
 * @param[in] ag Gaussian normalisation.
 * @param[in] aw Gaussian FWHM in radians.
 * @param[in,out] image Beam pattern data to which to apply antenna response.
 */
__global__
void oskar_cudak_antenna_gaussian_f(const int ns, const float* se,
        float ag, float aw, float2* image);

/**
 * @brief
 * CUDA kernel to apply sin(elevation) antenna response to beam pattern
 * (single precision).
 *
 * @details
 * This kernel multiplies the complex beam pattern by the sine of the pixel
 * elevations.
 *
 * @param[in] ns Number of pixels (source positions).
 * @param[in] se Source elevation positions in radians.
 * @param[in,out] image Beam pattern data to which to apply antenna response.
 */
__global__
void oskar_cudak_antenna_sine_f(const int ns, const float* se, float2* image);

/**
 * @brief
 * CUDA kernel to apply sin^2(elevation) antenna response to beam pattern
 * (single precision).
 *
 * @details
 * This kernel multiplies the complex beam pattern by the square of the sine
 * of the pixel elevations.
 *
 * @param[in] ns Number of pixels (source positions).
 * @param[in] se Source elevation positions in radians.
 * @param[in,out] image Beam pattern data to which to apply antenna response.
 */
__global__
void oskar_cudak_antenna_sine_squared_f(const int ns, const float* se,
        float2* image);

/**
 * @brief
 * CUDA kernel to apply Gaussian antenna response to beam pattern
 * (double precision).
 *
 * @details
 * This kernel multiplies the complex beam pattern by a Gaussian of a size
 * corresponding to the given parameters.
 *
 * @param[in] ns Number of pixels (source positions).
 * @param[in] se Source elevation positions in radians.
 * @param[in] ag Gaussian normalisation.
 * @param[in] aw Gaussian FWHM in radians.
 * @param[in,out] image Beam pattern data to which to apply antenna response.
 */
__global__
void oskar_cudak_antenna_gaussian_d(const int ns, const double* se,
        double ag, double aw, double2* image);

/**
 * @brief
 * CUDA kernel to apply sin(elevation) antenna response to beam pattern
 * (double precision).
 *
 * @details
 * This kernel multiplies the complex beam pattern by the sine of the pixel
 * elevations.
 *
 * @param[in] ns Number of pixels (source positions).
 * @param[in] se Source elevation positions in radians.
 * @param[in,out] image Beam pattern data to which to apply antenna response.
 */
__global__
void oskar_cudak_antenna_sine_d(const int ns, const double* se,
        double2* image);

/**
 * @brief
 * CUDA kernel to apply sin^2(elevation) antenna response to beam pattern
 * (double precision).
 *
 * @details
 * This kernel multiplies the complex beam pattern by the square of the sine
 * of the pixel elevations.
 *
 * @param[in] ns Number of pixels (source positions).
 * @param[in] se Source elevation positions in radians.
 * @param[in,out] image Beam pattern data to which to apply antenna response.
 */
__global__
void oskar_cudak_antenna_sine_squared_d(const int ns, const double* se,
        double2* image);

#endif // OSKAR_BF_CUDAK_ANTENNA_H_
