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

#include "station/cudak/oskar_cudak_phi_theta_to_normalised.h"
#include "utility/oskar_vector_types.h"

// Single precision.
__global__
void oskar_cudak_phi_theta_to_normalised_f(const int n, const float* phi,
        const float* theta, const float min_phi, const float min_theta,
        const float range_phi, const float range_theta, float* norm_phi,
        float* norm_theta)
{
    // Array element ID.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Get input coordinates.
    float c_phi = phi[i];
    float c_theta = theta[i];

    // Re-scale input coordinates.
    c_phi = (c_phi - min_phi) / range_phi;
    c_theta = (c_theta - min_theta) / range_theta;

    // Copy to output arrays.
    norm_phi[i] = c_phi;
    norm_theta[i] = c_theta;
}

// Double precision.
__global__
void oskar_cudak_phi_theta_to_normalised_d(const int n, const double* phi,
        const double* theta, const double min_phi, const double min_theta,
        const double range_phi, const double range_theta, double* norm_phi,
        double* norm_theta)
{
    // Array element ID.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Get input coordinates.
    double c_phi = phi[i];
    double c_theta = theta[i];

    // Re-scale input coordinates.
    c_phi = (c_phi - min_phi) / range_phi;
    c_theta = (c_theta - min_theta) / range_theta;

    // Copy to output arrays.
    norm_phi[i] = c_phi;
    norm_theta[i] = c_theta;
}
