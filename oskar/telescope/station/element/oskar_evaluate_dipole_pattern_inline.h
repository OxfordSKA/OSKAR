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

/*
 * Algorithm is as follows.
 *
 * Start with original expression for dipole pattern, with theta_prime the
 * polar angle from the dipole axis:
 *
 * E_theta_prime = (cos(kL * cos(theta_prime)) - cos_kL) / sin(theta_prime)
 * (E_phi_prime = 0)
 *
 * Substitute theta_prime = acos(cos_phi * sin_theta):
 *
 * E_theta_prime = (cos(kL * cos_phi * sin_theta) - cos_kL) /
 *                   sqrt(1 - cos_phi*cos_phi * sin_theta*sin_theta)
 *
 * Then rotate E_theta_prime vector to express as components along
 * the required E_theta and E_phi directions. This is performed using the
 * difference of bearing angles between the required point (start) and the
 * end of the dipole axis (finish):
 *
 * beta = atan2(-sin_phi, -cos_theta * cos_phi)
 * E_theta = cos(beta) * E_theta_prime; E_phi = -sin(beta) * E_theta_prime;
 *
 * (The same result can be achieved by projecting E_theta_prime along both the
 * E_theta and E_phi directions, but this is more efficient.)
 *
 * Simplifying gives:
 *
 * E_theta = -cos_phi * cos_theta * (cos(kL * cos_phi * sin_theta) - cos_kL)
 *              / (1 + cos_phi*cos_phi * (cos_theta*cos_theta - 1))
 * E_phi   = sin_phi * (cos(kL * cos_phi * sin_theta) - cos_kL)
 *              / (1 + cos_phi*cos_phi * (cos_theta*cos_theta - 1))
 */


/**
 * @brief Evaluates ideal dipole pattern at a single point (single precision).
 *
 * @details
 * This inline function evaluates the response of an ideal dipole at a single
 * point. The dipole is oriented along the x-axis. The angle phi is the
 * azimuthal angle from the x-axis towards the y-axis, and the angle theta is
 * the polar angle from the z-axis to the xy-plane.
 *
 * \f{eqnarray*){
 * E_{\theta} &=&
 *      \frac{-\cos\phi \cos\theta (\cos(k L \cos\phi \sin\theta) - \cos(k L))}
 *              {1 + \cos^2\phi (\cos^2\theta - 1)}
 * E_{\phi} &=&
 *      \frac{\sin\phi (\cos(k L \cos\phi \sin\theta) - \cos(k L))}
 *              {1 + \cos^2\phi (\cos^2\theta - 1)}
 * \f}
 *
 * @param[in] theta     Polar angle in radians.
 * @param[in] phi       Azimuthal angle in radians.
 * @param[in] kL        Wavenumber multiplied by dipole length in metres.
 * @param[in] cos_kL    Cosine of \p kL.
 * @param[out] E_theta  Complex dipole response in theta direction.
 * @param[out] E_phi    Complex dipole response in phi direction.
 */
OSKAR_INLINE
void oskar_evaluate_dipole_pattern_inline_f(const float theta,
        const float phi, const float kL, const float cos_kL,
        float2* E_theta, float2* E_phi)
{
    float sin_phi, cos_phi, sin_theta, cos_theta, t, denom;

#ifdef __CUDACC__
    sincosf(theta, &sin_theta, &cos_theta);
    sincosf(phi, &sin_phi, &cos_phi);
#else
    sin_theta = sinf(theta);
    cos_theta = cosf(theta);
    sin_phi = sinf(phi);
    cos_phi = cosf(phi);
#endif
    denom = 1.0f + cos_phi*cos_phi * (cos_theta*cos_theta - 1.0f);
    if (denom == 0.0f)
    {
        /* Early return if point is precisely at either end of the dipole. */
        E_theta->x = 0.0f;
        E_theta->y = 0.0f;
        E_phi->x = 0.0f;
        E_phi->y = 0.0f;
        return;
    }
    t = (cosf(kL * cos_phi * sin_theta) - cos_kL) / denom;

    /* Store real and imaginary components of E_theta, E_phi vectors. */
    E_theta->x = -cos_phi * cos_theta * t;
    E_theta->y = 0.0f;
    E_phi->x = sin_phi * t;
    E_phi->y = 0.0f;
}

/**
 * @brief Evaluates ideal dipole pattern at a single point (double precision).
 *
 * @details
 * This inline function evaluates the response of an ideal dipole at a single
 * point. The dipole is oriented along the x-axis. The angle phi is the
 * azimuthal angle from the x-axis towards the y-axis, and the angle theta is
 * the polar angle from the z-axis to the xy-plane.
 *
 * \f{eqnarray*){
 * E_{\theta} &=&
 *      \frac{-\cos\phi \cos\theta (\cos(k L \cos\phi \sin\theta) - \cos(k L))}
 *              {1 + \cos^2\phi (\cos^2\theta - 1)}
 * E_{\phi} &=&
 *      \frac{\sin\phi (\cos(k L \cos\phi \sin\theta) - \cos(k L))}
 *              {1 + \cos^2\phi (\cos^2\theta - 1)}
 * \f}
 *
 * @param[in] theta     Polar angle in radians.
 * @param[in] phi       Azimuthal angle in radians.
 * @param[in] kL        Wavenumber multiplied by dipole length in metres.
 * @param[in] cos_kL    Cosine of \p kL.
 * @param[out] E_theta  Complex dipole response in theta direction.
 * @param[out] E_phi    Complex dipole response in phi direction.
 */
OSKAR_INLINE
void oskar_evaluate_dipole_pattern_inline_d(const double theta,
        const double phi, const double kL, const double cos_kL,
        double2* E_theta, double2* E_phi)
{
    double sin_phi, cos_phi, sin_theta, cos_theta, t, denom;

#ifdef __CUDACC__
    sincos(theta, &sin_theta, &cos_theta);
    sincos(phi, &sin_phi, &cos_phi);
#else
    sin_theta = sin(theta);
    cos_theta = cos(theta);
    sin_phi = sin(phi);
    cos_phi = cos(phi);
#endif
    denom = 1.0 + cos_phi*cos_phi * (cos_theta*cos_theta - 1.0);
    if (denom == 0.0)
    {
        /* Early return if point is precisely at either end of the dipole. */
        E_theta->x = 0.0;
        E_theta->y = 0.0;
        E_phi->x = 0.0;
        E_phi->y = 0.0;
        return;
    }
    t = (cos(kL * cos_phi * sin_theta) - cos_kL) / denom;

    /* Store real and imaginary components of E_theta, E_phi vectors. */
    E_theta->x = -cos_phi * cos_theta * t;
    E_theta->y = 0.0;
    E_phi->x = sin_phi * t;
    E_phi->y = 0.0;
}

#ifdef __cplusplus
}
#endif
