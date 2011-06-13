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

#include "math/cudak/oskar_math_cudak_sph_to_lm.h"

// Single precision.

__global__
void oskar_math_cudakf_sph_to_lm(const int n, const float* lambda,
        const float* phi, const float lambda0, const float cosPhi0,
        const float sinPhi0, float* l, float* m)
{
    // Get the position ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;

    // Copy the input data from global memory.
    float cosPhi, sinPhi, sinLambda, cosLambda, relLambda, pphi;
    if (s < ns)
    {
        relLambda = lambda[s];
        pphi = phi[s];
    }
    __syncthreads(); // Coalesce memory accesses.

    // Convert from spherical to tangent-plane.
    relLambda -= lambda0;
    sincosf(relLambda, &sinLambda, &cosLambda);
    sincosf(pphi, &sinPhi, &cosPhi);
    float ll = cosPhi * sinLambda;
    float mm = cosPhi0 * sinPhi;
    mm -= sinPhi0 * cosPhi * cosLambda;

    // Output data.
    __syncthreads(); // Coalesce memory accesses.
    if (s < ns)
    {
        l[s] = ll;
        m[s] = mm;
    }
}

// Double precision.

__global__
void oskar_math_cudakd_sph_to_lm(const int n, const double* lambda,
        const double* phi, const double lambda0, const double cosPhi0,
        const double sinPhi0, double* l, double* m)
{
    // Get the position ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;

    // Copy the input data from global memory.
    double cosPhi, sinPhi, sinLambda, cosLambda, relLambda, pphi;
    if (s < ns)
    {
        relLambda = lambda[s];
        pphi = phi[s];
    }
    __syncthreads(); // Coalesce memory accesses.

    // Convert from spherical to tangent-plane.
    relLambda -= lambda0;
    sincos(relLambda, &sinLambda, &cosLambda);
    sincos(pphi, &sinPhi, &cosPhi);
    double ll = cosPhi * sinLambda;
    double mm = cosPhi0 * sinPhi;
    mm -= sinPhi0 * cosPhi * cosLambda;

    // Output data.
    __syncthreads(); // Coalesce memory accesses.
    if (s < ns)
    {
        l[s] = ll;
        m[s] = mm;
    }
}
