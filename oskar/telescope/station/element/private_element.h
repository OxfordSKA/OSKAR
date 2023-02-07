/*
 * Copyright (c) 2012-2023, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_PRIVATE_ELEMENT_H_
#define OSKAR_PRIVATE_ELEMENT_H_

#include <mem/oskar_mem.h>
#include <splines/oskar_splines.h>

struct oskar_Element
{
    int precision, mem_location;
    int element_type; /* Dipole or isotropic. */
    int taper_type; /* Tapering type. */
    int dipole_length_units; /* Units of dipole length (metres or wavelengths). */
    double dipole_length; /* Length of dipole. */
    double cosine_power; /* For a cosine taper, the power of the cosine. */
    double gaussian_fwhm_rad; /* For a Gaussian taper, the FWHM in radians. */

    /* Data for numerically-defined element patterns. */
    /* The arrays of fitted data are per-frequency. */
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
    oskar_Mem **sph_wave_feko;
    oskar_Mem **sph_wave_galileo;
};

#ifndef OSKAR_ELEMENT_TYPEDEF_
#define OSKAR_ELEMENT_TYPEDEF_
typedef struct oskar_Element oskar_Element;
#endif /* OSKAR_ELEMENT_TYPEDEF_ */

#endif /* OSKAR_PRIVATE_ELEMENT_H_ */
