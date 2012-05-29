/*
 * Copyright (c) 2012, The University of Oxford
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

#ifndef OSKAR_ELEMENT_MODEL_H_
#define OSKAR_ELEMENT_MODEL_H_

/**
 * @file oskar_ElementModel.h
 */

#include "math/oskar_SplineData.h"
#include "utility/oskar_Mem.h"

/**
 * @brief Structure to hold antenna (embedded element) pattern data.
 *
 * @details
 * This structure holds the spline coefficients and knot positions for
 * both polarisations of the antenna element.
 */
struct oskar_ElementModel
{
    int polarised; /**< True if element is polarised; false if point-like. */
    int tapering_function; /**< Type of tapering function on an ideal dipole. */
    int zenith_relative; /**< True if coordinates are relative to the zenith;
                              false if relative to the phase centre. */
    double gaussian_fwhm_deg; /**< For a Gaussian taper, the FWHM in degrees. */
    oskar_Mem filename_x;
    oskar_Mem filename_y;
    oskar_SplineData theta_re_x;
    oskar_SplineData theta_im_x;
    oskar_SplineData phi_re_x;
    oskar_SplineData phi_im_x;
    oskar_SplineData theta_re_y;
    oskar_SplineData theta_im_y;
    oskar_SplineData phi_re_y;
    oskar_SplineData phi_im_y;
};
typedef struct oskar_ElementModel oskar_ElementModel;

enum {
    OSKAR_ELEMENT_TAPER_NONE = 0,
    OSKAR_ELEMENT_TAPER_COS = 1,
    OSKAR_ELEMENT_TAPER_COS_SQUARED = 2,
    OSKAR_ELEMENT_TAPER_GAUSSIAN = 3
};

#endif /* OSKAR_ELEMENT_MODEL_H_ */
