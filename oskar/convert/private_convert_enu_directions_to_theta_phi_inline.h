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

#ifndef OSKAR_CONVERT_ENU_DIRECTIONS_TO_THETA_PHI_INLINE_H_
#define OSKAR_CONVERT_ENU_DIRECTIONS_TO_THETA_PHI_INLINE_H_

#include <oskar_global.h>
#include <oskar_cmath.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
OSKAR_INLINE
void oskar_convert_enu_directions_to_theta_phi_inline_f(
        float x, float y, const float z, const float delta_phi,
        float* theta, float* phi)
{
    float p;

    /* Cartesian to spherical (with orientation offset). */
    p = atan2f(y, x) + delta_phi;
    p = fmodf(p, 2.0f * M_PIf);
    x = sqrtf(x*x + y*y);
    y = atan2f(x, z); /* Theta. */
    if (p < 0.0f) p += 2.0f * M_PIf; /* Get phi in range 0 to 2 pi. */
    *phi = p;
    *theta = y;
}

/* Double precision. */
OSKAR_INLINE
void oskar_convert_enu_directions_to_theta_phi_inline_d(
        double x, double y, const double z, const double delta_phi,
        double* theta, double* phi)
{
    double p;

    /* Cartesian to spherical (with orientation offset). */
    p = atan2(y, x) + delta_phi;
    p = fmod(p, 2.0 * M_PI);
    x = sqrt(x*x + y*y);
    y = atan2(x, z); /* Theta. */
    if (p < 0.0) p += 2.0 * M_PI; /* Get phi in range 0 to 2 pi. */
    *phi = p;
    *theta = y;
}

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_ENU_DIRECTIONS_TO_THETA_PHI_INLINE_H_ */
