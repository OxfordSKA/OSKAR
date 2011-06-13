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

#ifndef OSKAR_MATH_CUDAK_SPH_TO_LM_H_
#define OSKAR_MATH_CUDAK_SPH_TO_LM_H_

/**
 * @file oskar_math_cudak_sph_to_lm.h
 */

#include "utility/oskar_util_cuda_eclipse.h"

/**
 * @brief
 * CUDA kernel to project spherical coordinates (single precision).
 *
 * @details
 * Projects spherical coordinates at the specified tangent point using the
 * orthographic tangent-plane projection.
 *
 * @param[in] n       Number of positions.
 * @param[in] lamda   Longitude positions in radians.
 * @param[in] phi     Latitude positions in radians.
 * @param[in] lambd0  Centre longitude position in radians.
 * @param[in] cosPhi0 Cosine of central latitude.
 * @param[in] sinPhi0 Sine of central latitude.
 * @param[out] l      Projected l-position.
 * @param[out] m      Projected m-position.
 */
__global__
void oskar_math_cudakf_sph_to_lm(const int n, const float* lambda,
        const float* phi, const float lambda0, const float cosPhi0,
        const float sinPhi0, float* l, float* m);

/**
 * @brief
 * CUDA kernel to project spherical coordinates (double precision).
 *
 * @details
 * Projects spherical coordinates at the specified tangent point using the
 * orthographic tangent-plane projection.
 *
 * @param[in] n       Number of positions.
 * @param[in] lamda   Longitude positions in radians.
 * @param[in] phi     Latitude positions in radians.
 * @param[in] lambd0  Centre longitude position in radians.
 * @param[in] cosPhi0 Cosine of central latitude.
 * @param[in] sinPhi0 Sine of central latitude.
 * @param[out] l      Projected l-position.
 * @param[out] m      Projected m-position.
 */
__global__
void oskar_math_cudakd_sph_to_lm(const int n, const float* lambda,
        const float* phi, const float lambda0, const float cosPhi0,
        const float sinPhi0, float* l, float* m);

#endif // OSKAR_MATH_CUDAK_SPH_TO_LM_H_
