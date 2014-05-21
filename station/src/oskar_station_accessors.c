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

#include <private_station.h>
#include <oskar_station_accessors.h>

#ifdef __cplusplus
extern "C" {
#endif


/* Data common to all station types. */

int oskar_station_precision(const oskar_Station* model)
{
    return model->precision;
}

int oskar_station_location(const oskar_Station* model)
{
    return model->location;
}

int oskar_station_type(const oskar_Station* model)
{
    return model->station_type;
}

int oskar_station_normalise_final_beam(const oskar_Station* model)
{
    return model->normalise_final_beam;
}

double oskar_station_longitude_rad(const oskar_Station* model)
{
    return model->longitude_rad;
}

double oskar_station_latitude_rad(const oskar_Station* model)
{
    return model->latitude_rad;
}

double oskar_station_altitude_m(const oskar_Station* model)
{
    return model->altitude_m;
}

double oskar_station_beam_longitude_rad(const oskar_Station* model)
{
    return model->beam_longitude_rad;
}

double oskar_station_beam_latitude_rad(const oskar_Station* model)
{
    return model->beam_latitude_rad;
}

int oskar_station_beam_coord_type(const oskar_Station* model)
{
    return model->beam_coord_type;
}

oskar_SystemNoiseModel* oskar_station_system_noise_model(oskar_Station* model)
{
    return model->noise;
}

const oskar_SystemNoiseModel* oskar_station_system_noise_model_const(
        const oskar_Station* model)
{
    return model->noise;
}


/* Data used only for Gaussian beam stations. */

double oskar_station_gaussian_beam_fwhm_rad(const oskar_Station* model)
{
    return model->gaussian_beam_fwhm_rad;
}


/* Data used only for aperture array stations. */

int oskar_station_identical_children(const oskar_Station* model)
{
    return model->identical_children;
}

int oskar_station_num_elements(const oskar_Station* model)
{
    return model->num_elements;
}

int oskar_station_num_element_types(const oskar_Station* model)
{
    return model->num_element_types;
}

int oskar_station_use_polarised_elements(const oskar_Station* model)
{
    return model->use_polarised_elements;
}

int oskar_station_normalise_array_pattern(const oskar_Station* model)
{
    return model->normalise_array_pattern;
}

int oskar_station_enable_array_pattern(const oskar_Station* model)
{
    return model->enable_array_pattern;
}

int oskar_station_common_element_orientation(const oskar_Station* model)
{
    return model->common_element_orientation;
}

int oskar_station_array_is_3d(const oskar_Station* model)
{
    return model->array_is_3d;
}

int oskar_station_apply_element_errors(const oskar_Station* model)
{
    return model->apply_element_errors;
}

int oskar_station_apply_element_weight(const oskar_Station* model)
{
    return model->apply_element_weight;
}

double oskar_station_nominal_element_orientation_x_rad(const oskar_Station* model)
{
    return model->nominal_orientation_x;
}

double oskar_station_nominal_element_orientation_y_rad(const oskar_Station* model)
{
    return model->nominal_orientation_y;
}

double oskar_station_element_orientation_x_rad(const oskar_Station* model,
        int index)
{
    const double *p;
    p = (const double*) oskar_mem_void_const(model->orientation_x_cpu);
    return p[index];
}

double oskar_station_element_orientation_y_rad(const oskar_Station* model,
        int index)
{
    const double *p;
    p = (const double*) oskar_mem_void_const(model->orientation_y_cpu);
    return p[index];
}

oskar_Mem* oskar_station_element_x_signal(oskar_Station* model)
{
    return model->x_signal;
}

const oskar_Mem* oskar_station_element_x_signal_const(const oskar_Station* model)
{
    return model->x_signal;
}

oskar_Mem* oskar_station_element_y_signal(oskar_Station* model)
{
    return model->y_signal;
}

const oskar_Mem* oskar_station_element_y_signal_const(const oskar_Station* model)
{
    return model->y_signal;
}

oskar_Mem* oskar_station_element_z_signal(oskar_Station* model)
{
    return model->z_signal;
}

const oskar_Mem* oskar_station_element_z_signal_const(const oskar_Station* model)
{
    return model->z_signal;
}

oskar_Mem* oskar_station_element_x_weights(oskar_Station* model)
{
    return model->x_weights;
}

const oskar_Mem* oskar_station_element_x_weights_const(const oskar_Station* model)
{
    return model->x_weights;
}

oskar_Mem* oskar_station_element_y_weights(oskar_Station* model)
{
    return model->y_weights;
}

const oskar_Mem* oskar_station_element_y_weights_const(const oskar_Station* model)
{
    return model->y_weights;
}

oskar_Mem* oskar_station_element_z_weights(oskar_Station* model)
{
    return model->z_weights;
}

const oskar_Mem* oskar_station_element_z_weights_const(const oskar_Station* model)
{
    return model->z_weights;
}

oskar_Mem* oskar_station_element_gain(oskar_Station* model)
{
    return model->gain;
}

const oskar_Mem* oskar_station_element_gain_const(const oskar_Station* model)
{
    return model->gain;
}

oskar_Mem* oskar_station_element_gain_error(oskar_Station* model)
{
    return model->gain_error;
}

const oskar_Mem* oskar_station_element_gain_error_const(const oskar_Station* model)
{
    return model->gain_error;
}

oskar_Mem* oskar_station_element_phase_offset(oskar_Station* model)
{
    return model->phase_offset;
}

const oskar_Mem* oskar_station_element_phase_offset_const(const oskar_Station* model)
{
    return model->phase_offset;
}

oskar_Mem* oskar_station_element_phase_error(oskar_Station* model)
{
    return model->phase_error;
}

const oskar_Mem* oskar_station_element_phase_error_const(const oskar_Station* model)
{
    return model->phase_error;
}

oskar_Mem* oskar_station_element_weight(oskar_Station* model)
{
    return model->weight;
}

const oskar_Mem* oskar_station_element_weight_const(const oskar_Station* model)
{
    return model->weight;
}

oskar_Mem* oskar_station_element_orientation_x_cpu(oskar_Station* model)
{
    return model->orientation_x_cpu;
}

const oskar_Mem* oskar_station_element_orientation_x_cpu_const(const oskar_Station* model)
{
    return model->orientation_x_cpu;
}

oskar_Mem* oskar_station_element_orientation_y_cpu(oskar_Station* model)
{
    return model->orientation_y_cpu;
}

const oskar_Mem* oskar_station_element_orientation_y_cpu_const(const oskar_Station* model)
{
    return model->orientation_y_cpu;
}

oskar_Mem* oskar_station_element_types(oskar_Station* model)
{
    return model->element_types;
}

const oskar_Mem* oskar_station_element_types_const(const oskar_Station* model)
{
    return model->element_types;
}

const int* oskar_station_element_types_cpu_const(const oskar_Station* model)
{
    return (const int*) oskar_mem_void_const(model->element_types_cpu);
}

int oskar_station_has_child(const oskar_Station* model)
{
    return model->child ? 1 : 0;
}

oskar_Station* oskar_station_child(oskar_Station* model, int i)
{
    return model->child[i];
}

const oskar_Station* oskar_station_child_const(const oskar_Station* model, int i)
{
    return model->child[i];
}

int oskar_station_has_element(const oskar_Station* model)
{
    return model->element_pattern ? 1 : 0;
}

oskar_Element* oskar_station_element(oskar_Station* model,
        int element_type_index)
{
    return model->element_pattern[element_type_index];
}

const oskar_Element* oskar_station_element_const(const oskar_Station* model,
        int element_type_index)
{
    return model->element_pattern[element_type_index];
}

int oskar_station_num_permitted_beams(const oskar_Station* model)
{
    return model->num_permitted_beams;
}

const oskar_Mem* oskar_station_permitted_beam_azimuth_rad_const(const oskar_Station* model)
{
    return model->permitted_beam_az;
}

const oskar_Mem* oskar_station_permitted_beam_elevation_rad_const(const oskar_Station* model)
{
    return model->permitted_beam_el;
}


/* Setters. */

void oskar_station_set_station_type(oskar_Station* model, int type)
{
    model->station_type = type;
}

void oskar_station_set_normalise_final_beam(oskar_Station* model, int value)
{
    model->normalise_final_beam = value;
}

void oskar_station_set_position(oskar_Station* model,
        double longitude_rad, double latitude_rad, double altitude_m)
{
    model->longitude_rad = longitude_rad;
    model->latitude_rad = latitude_rad;
    model->altitude_m = altitude_m;
}

void oskar_station_set_phase_centre(oskar_Station* model,
        int beam_coord_type, double beam_longitude_rad,
        double beam_latitude_rad)
{
    model->beam_coord_type = beam_coord_type;
    model->beam_longitude_rad = beam_longitude_rad;
    model->beam_latitude_rad = beam_latitude_rad;
}

void oskar_station_set_gaussian_beam_fwhm_rad(oskar_Station* model,
        double value)
{
    model->gaussian_beam_fwhm_rad = value;
}

void oskar_station_set_use_polarised_elements(oskar_Station* model, int value)
{
    model->use_polarised_elements = value;
}

void oskar_station_set_normalise_array_pattern(oskar_Station* model, int value)
{
    model->normalise_array_pattern = value;
}

void oskar_station_set_enable_array_pattern(oskar_Station* model, int value)
{
    model->enable_array_pattern = value;
}

#ifdef __cplusplus
}
#endif
