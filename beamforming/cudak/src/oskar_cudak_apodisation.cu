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


#include "beamforming/cudak/oskar_cudak_apodisation.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Note: Each thread applies the apodisation to one antenna for all beams.

// Single precision kernels.

__global__
void oskar_cudak_apodisation_hann_f(const int na, const float* ax,
        const float* ay, const int nb, const float fwhm, float2* weights)
{
    // Antenna index being processed by the thread.
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a > na) return;

    // Calculate Hann weight for the given antenna radius.
    const float r = sqrtf(ax[a] * ax[a] + ay[a] * ay[a]);
    const float weight = 0.5 * (1 + cosf((M_PI * r) / fwhm));

    // Loop over beam directions for the antenna and apply apodisation.
    for (int b = 0; b < nb; ++b)
    {
        const int idx = b * na + a;
        weights[idx].x *= weight;
        weights[idx].y *= weight;
    }
}

// Double precision kernels.

__global__
void oskar_cudak_apodisation_hann_d(const int na, const double* ax,
        const double* ay, const int nb, const double fwhm, double2* weights)
{
    // Antenna index being processed by the thread.
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a > na) return;

    // Calculate Hann weight for the given antenna radius.
    const double r = sqrt(ax[a] * ax[a] + ay[a] * ay[a]);
    const double weight = 0.5 * (1 + cos((M_PI * r) / fwhm));

    // Loop over beam directions for the antenna and apply apodisation.
    for (int b = 0; b < nb; ++b)
    {
        const int idx = b * na + a;
        weights[idx].x *= weight;
        weights[idx].y *= weight;
    }
}
