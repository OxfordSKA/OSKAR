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

#include "station/cudak/oskar_cudak_evaluate_dipole_pattern.h"
#include "station/cudak/oskar_cudaf_hor_lmn_to_az_el.h"
#include <math.h>

// Single precision.
__global__
void oskar_cudak_evaluate_dipole_pattern_f(int num_sources, float* l,
        float* m, float* n, float orientation_x, float orientation_y,
        float4c* pattern)
{
    // Source index being processed by the thread.
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s > num_sources) return;

    // Convert direction cosines to azimuth/elevation.
    // Angle alpha is the angle of the source from the axis of the dipole.
    float az, el, alpha, Ex, Ey;
    oskar_cudaf_hor_lmn_to_az_el_f(l[s], m[s], n[s], &az, &el);

    // Evaluate ideal dipole beam for antenna nominally in direction of x.
    alpha = (orientation_x - 0.5 * M_PI) + az;
    Ex = sinf(alpha);

    // Evaluate ideal dipole beam for antenna nominally in direction of y.
    alpha = orientation_y + az;
    Ey = sinf(alpha);

    // Store components.
    pattern[s].a.x = 0.0;
    pattern[s].a.y = -Ex;
    pattern[s].b.x = 0.0;
    pattern[s].b.y = 0.0;
    pattern[s].c.x = 0.0;
    pattern[s].c.y = -Ey;
    pattern[s].d.x = 0.0;
    pattern[s].d.y = 0.0;
}

// Double precision.
__global__
void oskar_cudak_evaluate_dipole_pattern_d(int num_sources, double* l,
        double* m, double* n, double orientation_x, double orientation_y,
        double4c* pattern)
{
    // Source index being processed by the thread.
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s > num_sources) return;

    // Convert direction cosines to azimuth/elevation.
    // Angle alpha is the angle of the source from the axis of the dipole.
    double az, el, alpha, Ex, Ey;
    oskar_cudaf_hor_lmn_to_az_el_d(l[s], m[s], n[s], &az, &el);

    // Evaluate ideal dipole beam for antenna nominally in direction of x.
    alpha = (orientation_x - 0.5 * M_PI) + az;
    Ex = sin(alpha);

    // Evaluate ideal dipole beam for antenna nominally in direction of y.
    alpha = orientation_y + az;
    Ey = sin(alpha);

    // Store components.
    pattern[s].a.x = 0.0;
    pattern[s].a.y = Ex;
    pattern[s].b.x = 0.0;
    pattern[s].b.y = 0.0;
    pattern[s].c.x = 0.0;
    pattern[s].c.y = Ey;
    pattern[s].d.x = 0.0;
    pattern[s].d.y = 0.0;
}
