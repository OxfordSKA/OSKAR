/*
 * Copyright (c) 2013-2023, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
int oskar_element_has_spherical_wave_feko_data(const oskar_Element* data,
        int freq_id);

OSKAR_EXPORT
int oskar_element_has_spherical_wave_galileo_data(const oskar_Element* data,
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
