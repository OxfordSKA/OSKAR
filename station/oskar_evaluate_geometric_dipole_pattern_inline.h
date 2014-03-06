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

#include <oskar_global.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
OSKAR_INLINE
void oskar_evaluate_geometric_dipole_pattern_inline_f(const float theta,
        const float phi, float2* E_theta, float2* E_phi)
{
    float sin_phi, cos_phi, cos_theta;

    /* Get source vector components, relative to a dipole along x. */
    cos_theta = cosf(theta);
#ifdef __CUDACC__
    sincosf(phi, &sin_phi, &cos_phi);
#else
    sin_phi = sinf(phi);
    cos_phi = cosf(phi);
#endif

    /* Store real and imaginary components of E_theta, E_phi vectors. */
    E_theta->x = cos_theta * cos_phi;
    E_theta->y = 0.0f;
    E_phi->x = -sin_phi;
    E_phi->y = 0.0f;
}

/* Double precision. */
OSKAR_INLINE
void oskar_evaluate_geometric_dipole_pattern_inline_d(const double theta,
        const double phi, double2* E_theta, double2* E_phi)
{
    double sin_phi, cos_phi, cos_theta;

    /* Get source vector components, relative to a dipole along x. */
    cos_theta = cos(theta);
#ifdef __CUDACC__
    sincos(phi, &sin_phi, &cos_phi);
#else
    sin_phi = sin(phi);
    cos_phi = cos(phi);
#endif

    /* Store real and imaginary components of E_theta, E_phi vectors. */
    E_theta->x = cos_theta * cos_phi;
    E_theta->y = 0.0;
    E_phi->x = -sin_phi;
    E_phi->y = 0.0;
}

#ifdef __cplusplus
}
#endif
