/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#ifndef OSKAR_PRIVATE_ELEMENT_H_
#define OSKAR_PRIVATE_ELEMENT_H_

/**
 * @file private_element.h
 */

#include <oskar_splines.h>
#include <oskar_mem.h>

/**
 * @brief Structure to hold antenna (embedded element) pattern data.
 *
 * @details
 * This structure holds the spline coefficients and knot positions for
 * both polarisations of the antenna element.
 */
struct oskar_Element
{
    int precision;
    int data_location;

    int element_type; /**< Dipole or isotropic. */
    int taper_type; /**< Tapering type. */
    double dipole_length_m; /**< Length of dipole in metres. */
    double cos_power; /**< For a cosine taper, the power of the cosine. */
    double gaussian_fwhm_rad; /**< For a Gaussian taper, the FWHM in radians. */
    oskar_Mem* filename_x;
    oskar_Mem* filename_y;
    oskar_Splines* theta_re_x;
    oskar_Splines* theta_im_x;
    oskar_Splines* phi_re_x;
    oskar_Splines* phi_im_x;
    oskar_Splines* theta_re_y;
    oskar_Splines* theta_im_y;
    oskar_Splines* phi_re_y;
    oskar_Splines* phi_im_y;
};

#ifndef OSKAR_ELEMENT_TYPEDEF_
#define OSKAR_ELEMENT_TYPEDEF_
typedef struct oskar_Element oskar_Element;
#endif /* OSKAR_ELEMENT_TYPEDEF_ */

#endif /* OSKAR_PRIVATE_ELEMENT_H_ */
