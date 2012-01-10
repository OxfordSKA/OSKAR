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

#ifndef OSKAR_SPHERICAL_SPLINE_DATA_H_
#define OSKAR_SPHERICAL_SPLINE_DATA_H_

/**
 * @file oskar_SphericalSplineData.h
 */

#include "utility/oskar_Mem.h"

/**
 * @brief Structure to hold complex, spherical spline data.
 *
 * @details
 * This structure holds the data required to construct a complex spherical
 * surface from splines.
 *
 * The knot position arrays should be of size e = 8 + sqrt(n), where n is the
 * number of scatter points used to define the surface. If storage space is
 * tight, then e = 8 + sqrt(n/2) may be sufficient.
 *
 * The coefficient arrays should be of size (e-4) * (e-4), where e is as above.
 */
struct oskar_SphericalSplineData
{
    int num_knots_theta_re;   /**< Number of knots in theta(real) direction. */
    int num_knots_theta_im;   /**< Number of knots in theta(imag) direction. */
    int num_knots_phi_re;     /**< Number of knots in phi(real) direction. */
    int num_knots_phi_im;     /**< Number of knots in phi(real) direction. */
    oskar_Mem knots_theta_re; /**< Knot positions in theta(real). */
    oskar_Mem knots_theta_im; /**< Knot positions in theta(imag). */
    oskar_Mem knots_phi_re;   /**< Knot positions in phi(real). */
    oskar_Mem knots_phi_im;   /**< Knot positions in phi(imag). */
    oskar_Mem coeff_re;       /**< Spline coefficient array (real). */
    oskar_Mem coeff_im;       /**< Spline coefficient array (imag). */
};

typedef struct oskar_SphericalSplineData oskar_SphericalSplineData;

#endif /* OSKAR_SPHERICAL_SPLINE_DATA_H_ */
