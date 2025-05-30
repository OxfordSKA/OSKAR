/*
 * Copyright (c) 2014-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
 * The supplied theta and phi_x positions of the sources are the <b>modified</b>
 * source positions. They must be adjusted relative to a dipole with its axis
 * oriented along the x-direction.
 * The phi_y angles would usually be 90 degrees larger, if the dipoles are
 * perpendicular.
 *
 * @param[in] num_points      Number of points.
 * @param[in] theta           Source theta values in rad.
 * @param[in] phi_x           Source phi values relative to X dipole in rad.
 * @param[in] phi_y           Source phi values relative to Y dipole in rad.
 * @param[in] freq_hz         Observing frequency in Hz.
 * @param[in] dipole_length_m Length of dipole in metres.
 * @param[in] swap_xy         If true, swap the X and Y dipole in the output.
 * @param[in] offset_out      Start offset into output array.
 * @param[out] pattern        Array of output Jones matrices/scalars per source.
 * @param[in,out] status      Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_dipole_pattern(
        int num_points,
        const oskar_Mem* theta,
        const oskar_Mem* phi_x,
        const oskar_Mem* phi_y,
        double freq_hz,
        double dipole_length_m,
        int swap_xy,
        int offset_out,
        oskar_Mem* pattern,
        int* status
);

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
