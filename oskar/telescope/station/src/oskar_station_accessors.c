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

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station_accessors.h"

#ifdef __cplusplus
extern "C" {
#endif


/* Data common to all station types. */

int oskar_station_unique_id(const oskar_Station* model)
{
    return model ? model->unique_id : 0;
}

int oskar_station_precision(const oskar_Station* model)
{
    return model ? model->precision : 0;
}

int oskar_station_mem_location(const oskar_Station* model)
{
    return model ? model->mem_location : 0;
}

int oskar_station_type(const oskar_Station* model)
{
    return model ? model->station_type : 0;
}

int oskar_station_normalise_final_beam(const oskar_Station* model)
{
    return model ? model->normalise_final_beam : 0;
}

double oskar_station_lon_rad(const oskar_Station* model)
{
    return model ? model->lon_rad : 0.0;
}

double oskar_station_lat_rad(const oskar_Station* model)
{
    return model ? model->lat_rad : 0.0;
}

double oskar_station_alt_metres(const oskar_Station* model)
{
    return model ? model->alt_metres : 0.0;
}

double oskar_station_polar_motion_x_rad(const oskar_Station* model)
{
    return model ? model->pm_x_rad : 0.0;
}

double oskar_station_polar_motion_y_rad(const oskar_Station* model)
{
    return model ? model->pm_y_rad : 0.0;
}

double oskar_station_beam_lon_rad(const oskar_Station* model)
{
    return model ? model->beam_lon_rad : 0.0;
}

double oskar_station_beam_lat_rad(const oskar_Station* model)
{
    return model ? model->beam_lat_rad : 0.0;
}

int oskar_station_beam_coord_type(const oskar_Station* model)
{
    return model ? model->beam_coord_type : 0;
}

oskar_Mem* oskar_station_noise_freq_hz(oskar_Station* model)
{
    return model ? model->noise_freq_hz : 0;
}

const oskar_Mem* oskar_station_noise_freq_hz_const(const oskar_Station* model)
{
    return model ? model->noise_freq_hz : 0;
}

oskar_Mem* oskar_station_noise_rms_jy(oskar_Station* model)
{
    return model ? model->noise_rms_jy : 0;
}

const oskar_Mem* oskar_station_noise_rms_jy_const(const oskar_Station* model)
{
    return model ? model->noise_rms_jy : 0;
}

/* Data used only for Gaussian beam stations. */

double oskar_station_gaussian_beam_fwhm_rad(const oskar_Station* model)
{
    return model ? model->gaussian_beam_fwhm_rad : 0.0;
}

double oskar_station_gaussian_beam_reference_freq_hz(const oskar_Station* model)
{
    return model ? model->gaussian_beam_reference_freq_hz : 0.0;
}

/* Data used only for aperture array stations. */

int oskar_station_identical_children(const oskar_Station* model)
{
    return model ? model->identical_children : 0;
}

int oskar_station_num_elements(const oskar_Station* model)
{
    return model ? model->num_elements : 0;
}

int oskar_station_num_element_types(const oskar_Station* model)
{
    return model ? model->num_element_types : 0;
}

int oskar_station_normalise_array_pattern(const oskar_Station* model)
{
    return model ? model->normalise_array_pattern : 0;
}

int oskar_station_enable_array_pattern(const oskar_Station* model)
{
    return model ? model->enable_array_pattern : 0;
}

int oskar_station_common_element_orientation(const oskar_Station* model)
{
    return model ? model->common_element_orientation : 0;
}

int oskar_station_array_is_3d(const oskar_Station* model)
{
    return model ? model->array_is_3d : 0;
}

int oskar_station_apply_element_errors(const oskar_Station* model)
{
    return model ? model->apply_element_errors : 0;
}

int oskar_station_apply_element_weight(const oskar_Station* model)
{
    return model ? model->apply_element_weight : 0;
}

unsigned int oskar_station_seed_time_variable_errors(const oskar_Station* model)
{
    return model ? model->seed_time_variable_errors : 0u;
}

double oskar_station_element_x_alpha_rad(const oskar_Station* model,
        int index)
{
    if (!model) return 0.0;
    return ((const double*)
            oskar_mem_void_const(model->element_x_alpha_cpu))[index];
}

double oskar_station_element_x_beta_rad(const oskar_Station* model,
        int index)
{
    if (!model) return 0.0;
    return ((const double*)
            oskar_mem_void_const(model->element_x_beta_cpu))[index];
}

double oskar_station_element_x_gamma_rad(const oskar_Station* model,
        int index)
{
    if (!model) return 0.0;
    return ((const double*)
            oskar_mem_void_const(model->element_x_gamma_cpu))[index];
}

double oskar_station_element_y_alpha_rad(const oskar_Station* model,
        int index)
{
    if (!model) return 0.0;
    return ((const double*)
            oskar_mem_void_const(model->element_y_alpha_cpu))[index];
}

double oskar_station_element_y_beta_rad(const oskar_Station* model,
        int index)
{
    if (!model) return 0.0;
    return ((const double*)
            oskar_mem_void_const(model->element_y_beta_cpu))[index];
}

double oskar_station_element_y_gamma_rad(const oskar_Station* model,
        int index)
{
    if (!model) return 0.0;
    return ((const double*)
            oskar_mem_void_const(model->element_y_gamma_cpu))[index];
}

oskar_Mem* oskar_station_element_true_x_enu_metres(oskar_Station* model)
{
    return model ? model->element_true_x_enu_metres : 0;
}

const oskar_Mem* oskar_station_element_true_x_enu_metres_const(
        const oskar_Station* model)
{
    return model ? model->element_true_x_enu_metres : 0;
}

oskar_Mem* oskar_station_element_true_y_enu_metres(oskar_Station* model)
{
    return model ? model->element_true_y_enu_metres : 0;
}

const oskar_Mem* oskar_station_element_true_y_enu_metres_const(
        const oskar_Station* model)
{
    return model ? model->element_true_y_enu_metres : 0;
}

oskar_Mem* oskar_station_element_true_z_enu_metres(oskar_Station* model)
{
    return model ? model->element_true_z_enu_metres : 0;
}

const oskar_Mem* oskar_station_element_true_z_enu_metres_const(
        const oskar_Station* model)
{
    return model ? model->element_true_z_enu_metres : 0;
}

oskar_Mem* oskar_station_element_measured_x_enu_metres(oskar_Station* model)
{
    return model ? model->element_measured_x_enu_metres : 0;
}

const oskar_Mem* oskar_station_element_measured_x_enu_metres_const(
        const oskar_Station* model)
{
    return model ? model->element_measured_x_enu_metres : 0;
}

oskar_Mem* oskar_station_element_measured_y_enu_metres(oskar_Station* model)
{
    return model ? model->element_measured_y_enu_metres : 0;
}

const oskar_Mem* oskar_station_element_measured_y_enu_metres_const(
        const oskar_Station* model)
{
    return model ? model->element_measured_y_enu_metres : 0;
}

oskar_Mem* oskar_station_element_measured_z_enu_metres(oskar_Station* model)
{
    return model ? model->element_measured_z_enu_metres : 0;
}

const oskar_Mem* oskar_station_element_measured_z_enu_metres_const(
        const oskar_Station* model)
{
    return model ? model->element_measured_z_enu_metres : 0;
}

oskar_Mem* oskar_station_element_cable_length_error_metres(
        oskar_Station* model)
{
    return model ? model->element_cable_length_error : 0;
}

const oskar_Mem* oskar_station_element_cable_length_error_metres_const(
        const oskar_Station* model)
{
    return model ? model->element_cable_length_error : 0;
}

oskar_Mem* oskar_station_element_gain(oskar_Station* model)
{
    return model ? model->element_gain : 0;
}

const oskar_Mem* oskar_station_element_gain_const(const oskar_Station* model)
{
    return model ? model->element_gain : 0;
}

oskar_Mem* oskar_station_element_gain_error(oskar_Station* model)
{
    return model ? model->element_gain_error : 0;
}

const oskar_Mem* oskar_station_element_gain_error_const(
        const oskar_Station* model)
{
    return model ? model->element_gain_error : 0;
}

oskar_Mem* oskar_station_element_phase_offset_rad(oskar_Station* model)
{
    return model ? model->element_phase_offset_rad : 0;
}

const oskar_Mem* oskar_station_element_phase_offset_rad_const(
        const oskar_Station* model)
{
    return model ? model->element_phase_offset_rad : 0;
}

oskar_Mem* oskar_station_element_phase_error_rad(oskar_Station* model)
{
    return model ? model->element_phase_error_rad : 0;
}

const oskar_Mem* oskar_station_element_phase_error_rad_const(
        const oskar_Station* model)
{
    return model ? model->element_phase_error_rad : 0;
}

oskar_Mem* oskar_station_element_weight(oskar_Station* model)
{
    return model ? model->element_weight : 0;
}

const oskar_Mem* oskar_station_element_weight_const(const oskar_Station* model)
{
    return model ? model->element_weight : 0;
}

oskar_Mem* oskar_station_element_types(oskar_Station* model)
{
    return model ? model->element_types : 0;
}

const oskar_Mem* oskar_station_element_types_const(const oskar_Station* model)
{
    return model ? model->element_types : 0;
}

const int* oskar_station_element_types_cpu_const(const oskar_Station* model)
{
    if (!model) return 0;
    return (const int*) oskar_mem_void_const(model->element_types_cpu);
}

const char* oskar_station_element_mount_types_const(const oskar_Station* model)
{
    if (!model) return 0;
    return oskar_mem_char_const(model->element_mount_types_cpu);
}

int oskar_station_has_child(const oskar_Station* model)
{
    return model ? (model->child ? 1 : 0) : 0;
}

oskar_Station* oskar_station_child(oskar_Station* model, int i)
{
    return model ? model->child[i] : 0;
}

const oskar_Station* oskar_station_child_const(const oskar_Station* model,
        int i)
{
    return model ? model->child[i] : 0;
}

int oskar_station_has_element(const oskar_Station* model)
{
    return model ? (model->element ? 1 : 0) : 0;
}

oskar_Element* oskar_station_element(oskar_Station* model,
        int element_type_index)
{
    return model ? model->element[element_type_index] : 0;
}

const oskar_Element* oskar_station_element_const(const oskar_Station* model,
        int element_type_index)
{
    return model ? model->element[element_type_index] : 0;
}

int oskar_station_num_permitted_beams(const oskar_Station* model)
{
    return model ? model->num_permitted_beams : 0;
}

const oskar_Mem* oskar_station_permitted_beam_az_rad_const(
        const oskar_Station* model)
{
    return model ? model->permitted_beam_az_rad : 0;
}

const oskar_Mem* oskar_station_permitted_beam_el_rad_const(
        const oskar_Station* model)
{
    return model ? model->permitted_beam_el_rad : 0;
}


/* Setters. */

void oskar_station_set_unique_ids(oskar_Station* model, int* counter)
{
    if (!model) return;
    model->unique_id = (*counter)++;
    if (model->child)
    {
        int i;
        for (i = 0; i < model->num_elements; ++i)
            oskar_station_set_unique_ids(model->child[i], counter);
    }
}

void oskar_station_set_station_type(oskar_Station* model, int type)
{
    if (!model) return;
    model->station_type = type;
}

void oskar_station_set_normalise_final_beam(oskar_Station* model, int value)
{
    if (!model) return;
    model->normalise_final_beam = value;
}

void oskar_station_set_position(oskar_Station* model,
        double longitude_rad, double latitude_rad, double altitude_m)
{
    if (!model) return;
    model->lon_rad = longitude_rad;
    model->lat_rad = latitude_rad;
    model->alt_metres = altitude_m;
}

void oskar_station_set_polar_motion(oskar_Station* model,
        double pm_x_rad, double pm_y_rad)
{
    int i;
    if (!model) return;
    model->pm_x_rad = pm_x_rad;
    model->pm_y_rad = pm_y_rad;

    /* Set recursively for all child stations. */
    if (oskar_station_has_child(model))
    {
        for (i = 0; i < model->num_elements; ++i)
        {
            oskar_station_set_polar_motion(model->child[i], pm_x_rad, pm_y_rad);
        }
    }
}

void oskar_station_set_phase_centre(oskar_Station* model,
        int beam_coord_type, double beam_longitude_rad,
        double beam_latitude_rad)
{
    if (!model) return;
    model->beam_coord_type = beam_coord_type;
    model->beam_lon_rad = beam_longitude_rad;
    model->beam_lat_rad = beam_latitude_rad;
}

void oskar_station_set_gaussian_beam_values(oskar_Station* model,
        double fwhm_rad, double ref_freq_hz)
{
    if (!model) return;
    model->gaussian_beam_fwhm_rad = fwhm_rad;
    model->gaussian_beam_reference_freq_hz = ref_freq_hz;
}

void oskar_station_set_normalise_array_pattern(oskar_Station* model, int value)
{
    if (!model) return;
    model->normalise_array_pattern = value;
}

void oskar_station_set_enable_array_pattern(oskar_Station* model, int value)
{
    if (!model) return;
    model->enable_array_pattern = value;
}

void oskar_station_set_seed_time_variable_errors(oskar_Station* model,
        unsigned int value)
{
    if (!model) return;
    model->seed_time_variable_errors = value;
}

#ifdef __cplusplus
}
#endif
