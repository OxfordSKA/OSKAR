/*
 * Copyright (c) 2014, The University of Oxford
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

#ifndef OSKAR_CONVERT_LUDWIG3_TO_THETA_PHI_COMPONENTS_CUDA_H_
#define OSKAR_CONVERT_LUDWIG3_TO_THETA_PHI_COMPONENTS_CUDA_H_

/**
 * @file oskar_convert_ludwig3_to_theta_phi_components_cuda.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Convert vector components from the Ludwig-3 system to the theta-phi system
 * (single precision).
 *
 * @details
 * This CUDA function converts vector components from the Ludwig-3 system (H/V)
 * to the normal spherical theta-phi system.
 *
 * Note that all pointers refer to device memory.
 *
 * @param[in] num_points     Length of all arrays.
 * @param[in,out] d_h_theta  On entry, the complex H component;
 *                           on exit, the complex theta component.
 * @param[in,out] d_v_phi    On entry, the complex V component;
 *                           on exit, the complex phi component.
 * @param[in] phi            The phi coordinates of all points, in radians.
 * @param[in] stride         The memory stride of the vector components
 *                           (this must be 1 if contiguous).
 */
OSKAR_EXPORT
void oskar_convert_ludwig3_to_theta_phi_components_cuda_f(int num_points,
        float2* d_h_theta, float2* d_v_phi, const float* d_phi, int stride);

/**
 * @brief
 * Convert vector components from the Ludwig-3 system to the theta-phi system
 * (double precision).
 *
 * @details
 * This CUDA function converts vector components from the Ludwig-3 system (H/V)
 * to the normal spherical theta-phi system.
 *
 * Note that all pointers refer to device memory.
 *
 * @param[in] num_points     Length of all arrays.
 * @param[in,out] d_h_theta  On entry, the complex H component;
 *                           on exit, the complex theta component.
 * @param[in,out] d_v_phi    On entry, the complex V component;
 *                           on exit, the complex phi component.
 * @param[in] phi            The phi coordinates of all points, in radians.
 * @param[in] stride         The memory stride of the vector components
 *                           (this must be 1 if contiguous).
 */
OSKAR_EXPORT
void oskar_convert_ludwig3_to_theta_phi_components_cuda_d(int num_points,
        double2* d_h_theta, double2* d_v_phi, const double* d_phi, int stride);

#ifdef __CUDACC__

/* Kernels. */

__global__
void oskar_convert_ludwig3_to_theta_phi_components_cudak_f(
        const int num_points, float2* h_theta, float2* v_phi,
        const float* __restrict__ phi, const int stride);

__global__
void oskar_convert_ludwig3_to_theta_phi_components_cudak_d(
        const int num_points, double2* h_theta, double2* v_phi,
        const double* __restrict__ phi, const int stride);

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_LUDWIG3_TO_THETA_PHI_COMPONENTS_CUDA_H_ */
