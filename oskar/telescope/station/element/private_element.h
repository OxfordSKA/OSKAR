/*
 * Copyright (c) 2012-2019, The University of Oxford
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

#include <mem/oskar_mem.h>
#include <splines/oskar_splines.h>

struct oskar_Element
{
    int precision, mem_location;
    int x_element_type,               y_element_type; /* Dipole or isotropic. */
    int x_taper_type,                 y_taper_type;
    int x_dipole_length_units,        y_dipole_length_units;
    double x_dipole_length,           y_dipole_length;
    double x_taper_cosine_power,      y_taper_cosine_power;
    double x_taper_gaussian_fwhm_rad, y_taper_gaussian_fwhm_rad;
    double x_taper_ref_freq_hz,       y_taper_ref_freq_hz;

    /* OLD: */
    int element_type; /* Dipole or isotropic. */
    int taper_type; /* Tapering type. */
    int dipole_length_units; /* Units of dipole length (metres or wavelengths). */
    double dipole_length; /* Length of dipole. */
    double cosine_power; /* For a cosine taper, the power of the cosine. */
    double gaussian_fwhm_rad; /* For a Gaussian taper, the FWHM in radians. */

    /* Data for numerically-defined element patterns. */
    /* The arrays of fitted data are per-frequency. */
    int coord_sys;
    double max_radius_rad;
    int num_freq;
    double* freqs_hz; /* Array of frequencies in Hz. */
    oskar_Mem **filename_x, **filename_y, **filename_scalar;

    /* Spline data. */
    oskar_Splines **x_h_re, **x_h_im;
    oskar_Splines **x_v_re, **x_v_im;
    oskar_Splines **y_h_re, **y_h_im;
    oskar_Splines **y_v_re, **y_v_im;
    oskar_Splines **scalar_re, **scalar_im;

    /* Spherical wave data. */
    int *common_phi_coords;
    int *l_max;
    oskar_Mem **sph_wave;
};

#ifndef OSKAR_ELEMENT_TYPEDEF_
#define OSKAR_ELEMENT_TYPEDEF_
typedef struct oskar_Element oskar_Element;
#endif /* OSKAR_ELEMENT_TYPEDEF_ */

#endif /* OSKAR_PRIVATE_ELEMENT_H_ */
