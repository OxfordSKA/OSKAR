/*
 * Copyright (c) 2014-2019, The University of Oxford
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

#ifndef OSKAR_EVALUATE_DIPOLE_PATTERN_H_
#define OSKAR_EVALUATE_DIPOLE_PATTERN_H_

/**
 * @file oskar_evaluate_dipole_pattern.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates pattern of a perfect dipole at source positions.
 *
 * @details
 * This function evaluates the pattern of a perfect dipole antenna
 * at the supplied source positions.
 *
 * The magnitude of the dipole response is given by
 *
 * \f[
 * E_{\theta^{'}} =
 *           \frac{\cos(\frac{kL}{2}\cos\phi\sin\theta) - \cos(\frac{kL}{2})}
 *                                 {\sqrt{1 - \cos^2\phi \sin^2\theta}};
 * \f]
 *
 * where phi and theta are the angles measured from x to y and from z to xy,
 * respectively.
 *
 * The supplied theta and phi positions of the sources are the <b>modified</b>
 * source positions. They must be adjusted relative to a dipole with its axis
 * oriented along the x-direction.
 *
 * @param[in] num_points         Number of points.
 * @param[in] theta              Point position (modified) theta values in rad.
 * @param[in] phi                Point position (modified) phi values in rad.
 * @param[in] freq_hz            Observing frequency in Hz.
 * @param[in] dipole_length_m    Length of dipole in metres.
 * @param[in] stride             Stride into output array (normally 1 or 4).
 * @param[in] offset             Start offset into output array.
 * @param[out] pattern           Array of output Jones matrices/scalars per source.
 * @param[in,out] status         Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_dipole_pattern(int num_points, const oskar_Mem* theta,
        const oskar_Mem* phi, double freq_hz, double dipole_length_m,
        int stride, int offset, oskar_Mem* pattern, int* status);

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
 *
 * The inline macro evaluates the response of an ideal dipole at a single
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
 */

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_DIPOLE_PATTERN_H_ */
