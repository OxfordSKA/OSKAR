/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#include <oskar_equation_of_equinoxes_fast.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DEG2RAD 0.0174532925199432957692
#define HOUR2RAD 0.261799387799149436539

double oskar_equation_of_equinoxes_fast(double mjd)
{
    double d, omega, L, delta_psi, epsilon, eqeq;

    /* Days from J2000.0. */
    d = mjd - 51544.5;

    /* Longitude of ascending node of the Moon. */
    omega = (125.04 - 0.052954 * d) * DEG2RAD;

    /* Mean Longitude of the Sun. */
    L = (280.47 + 0.98565 * d) * DEG2RAD;

    /* eqeq = delta_psi * cos(epsilon). */
    delta_psi = -0.000319 * sin(omega) - 0.000024 * sin(2.0 * L);
    epsilon = (23.4393 - 0.0000004 * d) * DEG2RAD;

    /* Return equation of equinoxes in radians. */
    eqeq = delta_psi * cos(epsilon) * HOUR2RAD;
    return eqeq;
}

#ifdef __cplusplus
}
#endif
