/*
 * Copyright (c) 2019, The University of Oxford
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
 * (Lines for small values of l must add trailing zeros.)
 *
 * @param[in,out] data         Pointer to element model data structure to fill.
 * @param[in]  filename        Data file name.
 * @param[in]  freq_hz         Frequency at which element data applies, in Hz.
 * @param[in,out] status       Status return code.
 */
OSKAR_EXPORT
void oskar_element_load_spherical_wave_coeff(oskar_Element* data,
        const char* filename, double freq_hz, int* num_tmp, double** tmp,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
