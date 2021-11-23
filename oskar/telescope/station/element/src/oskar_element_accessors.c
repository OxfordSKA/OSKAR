/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/element/private_element.h"
#include "telescope/station/element/oskar_element.h"

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif


int oskar_element_precision(const oskar_Element* data)
{
    return data->precision;
}

int oskar_element_mem_location(const oskar_Element* data)
{
    return data->mem_location;
}

int oskar_element_has_x_spline_data(const oskar_Element* data,
        int freq_id)
{
    return (data->num_freq > freq_id) && ( /* Safe short-circuit. */
            oskar_splines_have_coeffs(data->x_h_re[freq_id]) ||
            oskar_splines_have_coeffs(data->x_h_im[freq_id]) ||
            oskar_splines_have_coeffs(data->x_v_re[freq_id]) ||
            oskar_splines_have_coeffs(data->x_v_im[freq_id]));
}

int oskar_element_has_y_spline_data(const oskar_Element* data,
        int freq_id)
{
    return (data->num_freq > freq_id) && ( /* Safe short-circuit. */
            oskar_splines_have_coeffs(data->y_h_re[freq_id]) ||
            oskar_splines_have_coeffs(data->y_h_im[freq_id]) ||
            oskar_splines_have_coeffs(data->y_v_re[freq_id]) ||
            oskar_splines_have_coeffs(data->y_v_im[freq_id]));
}

int oskar_element_has_scalar_spline_data(const oskar_Element* data,
        int freq_id)
{
    return (data->num_freq > freq_id) && ( /* Safe short-circuit. */
            oskar_splines_have_coeffs(data->scalar_re[freq_id]) ||
            oskar_splines_have_coeffs(data->scalar_im[freq_id]));
}

int oskar_element_has_spherical_wave_data(const oskar_Element* data,
        int freq_id)
{
    return (data->num_freq > freq_id) && ( /* Safe short-circuit. */
            data->l_max[freq_id] > 0);
}

int oskar_element_num_freq(const oskar_Element* data)
{
    return data->num_freq;
}

const double* oskar_element_freqs_hz_const(const oskar_Element* data)
{
    return data->freqs_hz;
}

int oskar_element_is_isotropic(const oskar_Element* data)
{
    return data->element_type == OSKAR_ELEMENT_TYPE_ISOTROPIC;
}

int oskar_element_taper_type(const oskar_Element* data)
{
    return data->taper_type;
}

double oskar_element_cosine_power(const oskar_Element* data)
{
    return data->cosine_power;
}

double oskar_element_gaussian_fwhm_rad(const oskar_Element* data)
{
    return data->gaussian_fwhm_rad;
}

double oskar_element_dipole_length(const oskar_Element* data)
{
    return data->dipole_length;
}

int oskar_element_dipole_length_units(const oskar_Element* data)
{
    return data->dipole_length_units;
}

const oskar_Mem* oskar_element_x_filename_const(const oskar_Element* data,
        int freq_id)
{
    return data->filename_x[freq_id];
}

const oskar_Mem* oskar_element_y_filename_const(const oskar_Element* data,
        int freq_id)
{
    return data->filename_y[freq_id];
}

const oskar_Mem* oskar_element_scalar_filename_const(const oskar_Element* data,
        int freq_id)
{
    return data->filename_scalar[freq_id];
}


/* Setters. */

void oskar_element_set_element_type(oskar_Element* data, const char* type,
        int* status)
{
    if (*status) return;
    if (!strncmp(type, "D", 1) || !strncmp(type, "d", 1))
    {
        data->element_type = OSKAR_ELEMENT_TYPE_DIPOLE;
    }
    else if (!strncmp(type, "I",  1) || !strncmp(type, "i",  1))
    {
        data->element_type = OSKAR_ELEMENT_TYPE_ISOTROPIC;
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
    }
}

void oskar_element_set_taper_type(oskar_Element* data, const char* type,
        int* status)
{
    if (*status) return;
    if (!strncmp(type, "N", 1) || !strncmp(type, "n", 1))
    {
        data->taper_type = OSKAR_ELEMENT_TAPER_NONE;
    }
    else if (!strncmp(type, "C",  1) || !strncmp(type, "c",  1))
    {
        data->taper_type = OSKAR_ELEMENT_TAPER_COSINE;
    }
    else if (!strncmp(type, "G",  1) || !strncmp(type, "g",  1))
    {
        data->taper_type = OSKAR_ELEMENT_TAPER_GAUSSIAN;
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
    }
}

void oskar_element_set_gaussian_fwhm_rad(oskar_Element* data, double value)
{
    data->gaussian_fwhm_rad = value;
}

void oskar_element_set_cosine_power(oskar_Element* data, double value)
{
    data->cosine_power = value;
}

void oskar_element_set_dipole_length(oskar_Element* data, double value,
        const char* units, int* status)
{
    if (*status) return;
    if (!strncmp(units, "W", 1) || !strncmp(units, "w", 1))
    {
        data->dipole_length_units = OSKAR_WAVELENGTHS;
    }
    else if (!strncmp(units, "M",  1) || !strncmp(units, "m",  1))
    {
        data->dipole_length_units = OSKAR_METRES;
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    data->dipole_length = value;
}

#ifdef __cplusplus
}
#endif
