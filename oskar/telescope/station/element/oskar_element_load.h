/*
 * Copyright (c) 2015-2016, The University of Oxford
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

#ifndef OSKAR_ELEMENT_LOAD_H_
#define OSKAR_ELEMENT_LOAD_H_

/**
 * @file oskar_element_load.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Loads element pattern parameters from a text file.
 *
 * @details
 * This function loads element pattern parameters from
 * a comma- or space-separated text file.
 *
 * The file may have one row containing the following columns,
 * in the following order:
 * - Character specifying the element pattern base type ('I' or 'D' for
 *   isotropic or dipole).
 * - If dipole, the length of the dipole.
 * - If dipole, the length units of the dipole ('M' or 'W' for
 *   metres or wavelengths).
 * - Character specifying tapering type ('C' or 'G' for cosine or Gaussian)
 * - Cosine power, or Gaussian FWHM, in degrees.
 * - Reference frequency for Gaussian FWHM value, in Hz.
 *
 * @param[out] element   Pointer to destination data structure to fill.
 * @param[in] filename   Name of the data file to load.
 * @param[in] x_pol      If set, load for x polarisation, else y polarisation.
 * @param[in,out] status Status return code.
 */
OSKAR_EXPORT
void oskar_element_load(oskar_Element* element, const char* filename,
        int x_pol, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_ELEMENT_LOAD_H_ */
