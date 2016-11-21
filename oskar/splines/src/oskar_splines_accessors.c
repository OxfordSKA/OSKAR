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

#include <private_splines.h>
#include <oskar_splines.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_splines_precision(const oskar_Splines* data)
{
    return data->precision;
}

int oskar_splines_mem_location(const oskar_Splines* data)
{
    return data->mem_location;
}

int oskar_splines_have_coeffs(const oskar_Splines* data)
{
    return (data->num_knots_x_theta) > 0 && (data->num_knots_y_phi > 0);
}

int oskar_splines_num_knots_x_theta(const oskar_Splines* data)
{
    return data->num_knots_x_theta;
}

int oskar_splines_num_knots_y_phi(const oskar_Splines* data)
{
    return data->num_knots_y_phi;
}

oskar_Mem* oskar_splines_knots_x(oskar_Splines* data)
{
    return data->knots_x_theta;
}

const oskar_Mem* oskar_splines_knots_x_theta_const(const oskar_Splines* data)
{
    return data->knots_x_theta;
}

oskar_Mem* oskar_splines_knots_y(oskar_Splines* data)
{
    return data->knots_y_phi;
}

const oskar_Mem* oskar_splines_knots_y_phi_const(const oskar_Splines* data)
{
    return data->knots_y_phi;
}

oskar_Mem* oskar_splines_coeff(oskar_Splines* data)
{
    return data->coeff;
}

const oskar_Mem* oskar_splines_coeff_const(const oskar_Splines* data)
{
    return data->coeff;
}

double oskar_splines_smoothing_factor(const oskar_Splines* data)
{
    return data->smoothing_factor;
}

#ifdef __cplusplus
}
#endif
