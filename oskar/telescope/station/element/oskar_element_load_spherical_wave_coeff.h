/*
 * Copyright (c) 2019-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_ELEMENT_LOAD_SPHERICAL_WAVE_COEFF_H_
#define OSKAR_ELEMENT_LOAD_SPHERICAL_WAVE_COEFF_H_

/**
 * @file oskar_element_load_spherical_wave_coeff.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Loads spherical wave coefficients from an ASCII text file.
 *
 * @details
 * This function loads spherical wave coefficients from a text file and
 * fills the provided data structure.
 *
 * The filename must contain sub-strings to specify which part of the
 * element pattern to load:
 *
 * - "_X_" for the X dipole;
 * - "_Y_" for the Y dipole;
 * - "_TE" for the transverse electric component;
 * - "_TM" for the transverse magnetic component;
 * - "_RE" for the real part;
 * - "_IM" for the imaginary part.
 *
 * Matching is done in both upper and lower case.
 *
 * The data file should contain whitespace- or comma-separated numerical
 * values and must contain enough coefficients to fully specify a spherical
 * wave/harmonic series up to an integer value of l_max.
 * The number of coefficients required for a given value of l_max
 * is (2 * l_max + 1) * l_max.
 *
 * @param[in,out] data         Pointer to element model data structure to fill.
 * @param[in]  filename        Data file name.
 * @param[in]  sph_wave_type   Specify type: 0=Original, 1=FEKO, 2=Galileo.
 * @param[in]  max_order       If greater than 0, the maximum order to load.
 * @param[in]  freq_hz         Frequency at which element data applies, in Hz.
 * @param[in,out] status       Status return code.
 */
OSKAR_EXPORT
void oskar_element_load_spherical_wave_coeff(
        oskar_Element* data,
        const char* filename,
        int sph_wave_type,
        int max_order,
        double freq_hz,
        int* num_tmp,
        double** tmp,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
