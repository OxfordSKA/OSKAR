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

#include "cuda/kernels/oskar_cudak_antenna.h"

#define PI_2_F 1.57079633f
#define PI_2_D 1.5707963267948966

// Single precision kernels.

__global__
void oskar_cudakf_antenna_gaussian(const int ns, const float* se,
        float ag, float aw, float2* image)
{
    // Pixel index being processed by the thread.
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s > ns) return;

    float zd2 = powf(PI_2_F - se[s], 2.0f); // Source zenith distance squared.
    float amp = 0.0f; // Prevent underflows (huge speed difference!).
    if (aw * zd2 < 30.0f)
        amp =  ag * expf(-zd2 * aw);

    image[s].x *= amp;
    image[s].y *= amp;
}

__global__
void oskar_cudakf_antenna_sine(const int ns, const float* se, float2* image)
{
    // Pixel index being processed by the thread.
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s > ns) return;

    float amp = sinf(se[s]); // Antenna response at pixel position.
    image[s].x *= amp;
    image[s].y *= amp;
}

__global__
void oskar_cudakf_antenna_sine_squared(const int ns, const float* se,
        float2* image)
{
    // Pixel index being processed by the thread.
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s > ns) return;

    float amp = powf(sinf(se[s]), 2.0f); // Antenna response at pixel position.
    image[s].x *= amp;
    image[s].y *= amp;
}

// Double precision kernels.

__global__
void oskar_cudakd_antenna_gaussian(const int ns, const double* se,
        double ag, double aw, double2* image)
{
    // Pixel index being processed by the thread.
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s > ns) return;

    double zd2 = pow(PI_2_D - se[s], 2.0); // Source zenith distance squared.
    double amp = 0.0; // Prevent underflows (huge speed difference!).
    if (aw * zd2 < 30.0)
        amp =  ag * exp(-zd2 * aw);

    image[s].x *= amp;
    image[s].y *= amp;
}

__global__
void oskar_cudakd_antenna_sine(const int ns, const double* se, double2* image)
{
    // Pixel index being processed by the thread.
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s > ns) return;

    double amp = sin(se[s]); // Antenna response at pixel position.
    image[s].x *= amp;
    image[s].y *= amp;
}

__global__
void oskar_cudakd_antenna_sine_squared(const int ns, const double* se,
        double2* image)
{
    // Pixel index being processed by the thread.
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s > ns) return;

    double amp = pow(sin(se[s]), 2.0); // Antenna response at pixel position.
    image[s].x *= amp;
    image[s].y *= amp;
}
