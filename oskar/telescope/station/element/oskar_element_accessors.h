/*
 * Copyright (c) 2013-2019, The University of Oxford
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

#ifndef OSKAR_ELEMENT_ACCESSORS_H_
#define OSKAR_ELEMENT_ACCESSORS_H_

/**
 * @file oskar_element_accessors.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_EXPORT
int oskar_element_precision(const oskar_Element* data);

OSKAR_EXPORT
int oskar_element_mem_location(const oskar_Element* data);

OSKAR_EXPORT
int oskar_element_has_x_spline_data(const oskar_Element* data,
        int freq_id);

OSKAR_EXPORT
int oskar_element_has_y_spline_data(const oskar_Element* data,
        int freq_id);

OSKAR_EXPORT
int oskar_element_has_scalar_spline_data(const oskar_Element* data,
        int freq_id);

OSKAR_EXPORT
int oskar_element_has_spherical_wave_data(const oskar_Element* data,
        int freq_id);

OSKAR_EXPORT
int oskar_element_num_freq(const oskar_Element* data);

OSKAR_EXPORT
const double* oskar_element_freqs_hz_const(const oskar_Element* data);

OSKAR_EXPORT
int oskar_element_is_isotropic(const oskar_Element* data);

OSKAR_EXPORT
int oskar_element_taper_type(const oskar_Element* data);

OSKAR_EXPORT
double oskar_element_cosine_power(const oskar_Element* data);

OSKAR_EXPORT
double oskar_element_gaussian_fwhm_rad(const oskar_Element* data);

OSKAR_EXPORT
double oskar_element_dipole_length(const oskar_Element* data);

OSKAR_EXPORT
int oskar_element_dipole_length_units(const oskar_Element* data);

OSKAR_EXPORT
const oskar_Mem* oskar_element_x_filename_const(const oskar_Element* data,
        int freq_id);

OSKAR_EXPORT
const oskar_Mem* oskar_element_y_filename_const(const oskar_Element* data,
        int freq_id);

OSKAR_EXPORT
const oskar_Mem* oskar_element_scalar_filename_const(const oskar_Element* data,
        int freq_id);


/* Setters. */

OSKAR_EXPORT
void oskar_element_set_element_type(oskar_Element* data, const char* type,
        int* status);

OSKAR_EXPORT
void oskar_element_set_taper_type(oskar_Element* data, const char* type,
        int* status);

OSKAR_EXPORT
void oskar_element_set_gaussian_fwhm_rad(oskar_Element* data, double value);

OSKAR_EXPORT
void oskar_element_set_cosine_power(oskar_Element* data, double value);

OSKAR_EXPORT
void oskar_element_set_dipole_length(oskar_Element* data, double value,
        const char* units, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_ELEMENT_ACCESSORS_H_ */
