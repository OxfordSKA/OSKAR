/*
 * Copyright (c) 2013-2014, The University of Oxford
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

int oskar_element_has_x_spline_data(const oskar_Element* data)
{
    return (data->num_freq > 0) && ( /* Short-circuit, so will be safe. */
            oskar_splines_have_coeffs(data->x_h_re[0]) ||
            oskar_splines_have_coeffs(data->x_h_im[0]) ||
            oskar_splines_have_coeffs(data->x_v_re[0]) ||
            oskar_splines_have_coeffs(data->x_v_im[0]));
}

int oskar_element_has_y_spline_data(const oskar_Element* data)
{
    return (data->num_freq > 0) && ( /* Short-circuit, so will be safe. */
            oskar_splines_have_coeffs(data->y_h_re[0]) ||
            oskar_splines_have_coeffs(data->y_h_im[0]) ||
            oskar_splines_have_coeffs(data->y_v_re[0]) ||
            oskar_splines_have_coeffs(data->y_v_im[0]));
}

int oskar_element_has_scalar_spline_data(const oskar_Element* data)
{
    return (data->num_freq > 0) && ( /* Short-circuit, so will be safe. */
            oskar_splines_have_coeffs(data->scalar_re[0]) ||
            oskar_splines_have_coeffs(data->scalar_im[0]));
}

int oskar_element_num_freq(const oskar_Element* data)
{
    return data->num_freq;
}

const double* oskar_element_freqs_hz_const(const oskar_Element* data)
{
    return data->freqs_hz;
}

int oskar_element_type(const oskar_Element* data)
{
    return data->element_type;
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

oskar_Mem* oskar_element_x_filename(oskar_Element* data, int freq_id)
{
    return data->filename_x[freq_id];
}

const oskar_Mem* oskar_element_x_filename_const(const oskar_Element* data,
        int freq_id)
{
    return data->filename_x[freq_id];
}

oskar_Mem* oskar_element_y_filename(oskar_Element* data, int freq_id)
{
    return data->filename_y[freq_id];
}

const oskar_Mem* oskar_element_y_filename_const(const oskar_Element* data,
        int freq_id)
{
    return data->filename_y[freq_id];
}

oskar_Mem* oskar_element_scalar_filename(oskar_Element* data, int freq_id)
{
    return data->filename_scalar[freq_id];
}

const oskar_Mem* oskar_element_scalar_filename_const(const oskar_Element* data,
        int freq_id)
{
    return data->filename_scalar[freq_id];
}



oskar_Splines* oskar_element_x_h_re(oskar_Element* data, int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->x_h_re[freq_id];
}

const oskar_Splines* oskar_element_x_h_re_const(const oskar_Element* data,
        int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->x_h_re[freq_id];
}

oskar_Splines* oskar_element_x_h_im(oskar_Element* data, int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->x_h_im[freq_id];
}

const oskar_Splines* oskar_element_x_h_im_const(const oskar_Element* data,
        int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->x_h_im[freq_id];
}

oskar_Splines* oskar_element_x_v_re(oskar_Element* data, int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->x_v_re[freq_id];
}

const oskar_Splines* oskar_element_x_v_re_const(const oskar_Element* data,
        int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->x_v_re[freq_id];
}

oskar_Splines* oskar_element_x_v_im(oskar_Element* data, int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->x_v_im[freq_id];
}

const oskar_Splines* oskar_element_x_v_im_const(const oskar_Element* data,
        int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->x_v_im[freq_id];
}



oskar_Splines* oskar_element_y_h_re(oskar_Element* data, int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->y_h_re[freq_id];
}

const oskar_Splines* oskar_element_y_h_re_const(const oskar_Element* data,
        int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->y_h_re[freq_id];
}

oskar_Splines* oskar_element_y_h_im(oskar_Element* data, int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->y_h_im[freq_id];
}

const oskar_Splines* oskar_element_y_h_im_const(const oskar_Element* data,
        int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->y_h_im[freq_id];
}

oskar_Splines* oskar_element_y_v_re(oskar_Element* data, int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->y_v_re[freq_id];
}

const oskar_Splines* oskar_element_y_v_re_const(const oskar_Element* data,
        int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->y_v_re[freq_id];
}

oskar_Splines* oskar_element_y_v_im(oskar_Element* data, int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->y_v_im[freq_id];
}

const oskar_Splines* oskar_element_y_v_im_const(const oskar_Element* data,
        int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->y_v_im[freq_id];
}

oskar_Splines* oskar_element_scalar_re(oskar_Element* data, int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->scalar_re[freq_id];
}

const oskar_Splines* oskar_element_scalar_re_const(const oskar_Element* data,
        int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->scalar_re[freq_id];
}

oskar_Splines* oskar_element_scalar_im(oskar_Element* data, int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->scalar_im[freq_id];
}

const oskar_Splines* oskar_element_scalar_im_const(const oskar_Element* data,
        int freq_id)
{
    if (freq_id >= data->num_freq) return 0;
    return data->scalar_im[freq_id];
}


/* Setters. */

void oskar_element_set_element_type(oskar_Element* data, const char* type,
        int* status)
{
    if (*status) return;
    if (!strncmp(type, "D", 1) || !strncmp(type, "d", 1))
        data->element_type = OSKAR_ELEMENT_TYPE_DIPOLE;
    else if (!strncmp(type, "G",  1) || !strncmp(type, "g",  1))
        data->element_type = OSKAR_ELEMENT_TYPE_GEOMETRIC_DIPOLE;
    else if (!strncmp(type, "I",  1) || !strncmp(type, "i",  1))
        data->element_type = OSKAR_ELEMENT_TYPE_ISOTROPIC;
    else
        *status = OSKAR_ERR_INVALID_ARGUMENT;
}

void oskar_element_set_taper_type(oskar_Element* data, const char* type,
        int* status)
{
    if (*status) return;
    if (!strncmp(type, "N", 1) || !strncmp(type, "n", 1))
        data->taper_type = OSKAR_ELEMENT_TAPER_NONE;
    else if (!strncmp(type, "C",  1) || !strncmp(type, "c",  1))
        data->taper_type = OSKAR_ELEMENT_TAPER_COSINE;
    else if (!strncmp(type, "G",  1) || !strncmp(type, "g",  1))
        data->taper_type = OSKAR_ELEMENT_TAPER_GAUSSIAN;
    else
        *status = OSKAR_ERR_INVALID_ARGUMENT;
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
        data->dipole_length_units = OSKAR_WAVELENGTHS;
    else if (!strncmp(units, "M",  1) || !strncmp(units, "m",  1))
        data->dipole_length_units = OSKAR_METRES;
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
