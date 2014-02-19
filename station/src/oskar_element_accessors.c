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

#include <private_element.h>
#include <oskar_element.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_element_type(const oskar_Element* data)
{
    return data->data_type;
}

int oskar_element_location(const oskar_Element* data)
{
    return data->data_location;
}

int oskar_element_has_spline_data(const oskar_Element* data)
{
    return oskar_splines_has_coeffs(data->theta_re_x);
}

int oskar_element_element_type(const oskar_Element* data)
{
    return data->element_type;
}

int oskar_element_taper_type(const oskar_Element* data)
{
    return data->taper_type;
}

double oskar_element_cos_power(const oskar_Element* data)
{
    return data->cos_power;
}

double oskar_element_gaussian_fwhm_rad(const oskar_Element* data)
{
    return data->gaussian_fwhm_rad;
}

oskar_Mem* oskar_element_x_filename(oskar_Element* data)
{
    return data->filename_x;
}

const oskar_Mem* oskar_element_x_filename_const(const oskar_Element* data)
{
    return data->filename_x;
}

oskar_Mem* oskar_element_y_filename(oskar_Element* data)
{
    return data->filename_y;
}

const oskar_Mem* oskar_element_y_filename_const(const oskar_Element* data)
{
    return data->filename_y;
}



oskar_Splines* oskar_element_x_theta_re(oskar_Element* data)
{
    return data->theta_re_x;
}

const oskar_Splines* oskar_element_x_theta_re_const(const oskar_Element* data)
{
    return data->theta_re_x;
}

oskar_Splines* oskar_element_x_theta_im(oskar_Element* data)
{
    return data->theta_im_x;
}

const oskar_Splines* oskar_element_x_theta_im_const(const oskar_Element* data)
{
    return data->theta_im_x;
}

oskar_Splines* oskar_element_x_phi_re(oskar_Element* data)
{
    return data->phi_re_x;
}

const oskar_Splines* oskar_element_x_phi_re_const(const oskar_Element* data)
{
    return data->phi_re_x;
}

oskar_Splines* oskar_element_x_phi_im(oskar_Element* data)
{
    return data->phi_im_x;
}

const oskar_Splines* oskar_element_x_phi_im_const(const oskar_Element* data)
{
    return data->phi_im_x;
}



oskar_Splines* oskar_element_y_theta_re(oskar_Element* data)
{
    return data->theta_re_y;
}

const oskar_Splines* oskar_element_y_theta_re_const(const oskar_Element* data)
{
    return data->theta_re_y;
}

oskar_Splines* oskar_element_y_theta_im(oskar_Element* data)
{
    return data->theta_im_y;
}

const oskar_Splines* oskar_element_y_theta_im_const(const oskar_Element* data)
{
    return data->theta_im_y;
}

oskar_Splines* oskar_element_y_phi_re(oskar_Element* data)
{
    return data->phi_re_y;
}

const oskar_Splines* oskar_element_y_phi_re_const(const oskar_Element* data)
{
    return data->phi_re_y;
}

oskar_Splines* oskar_element_y_phi_im(oskar_Element* data)
{
    return data->phi_im_y;
}

const oskar_Splines* oskar_element_y_phi_im_const(const oskar_Element* data)
{
    return data->phi_im_y;
}


/* Setters. */

void oskar_element_set_element_type(oskar_Element* data, int type)
{
    data->element_type = type;
}

void oskar_element_set_taper_type(oskar_Element* data, int type)
{
    data->taper_type = type;
}

void oskar_element_set_gaussian_fwhm_rad(oskar_Element* data, double value)
{
    data->gaussian_fwhm_rad = value;
}

void oskar_element_set_cos_power(oskar_Element* data, double value)
{
    data->cos_power = value;
}

#ifdef __cplusplus
}
#endif
