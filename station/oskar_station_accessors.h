/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#ifndef OSKAR_STATION_ACCESSORS_H_
#define OSKAR_STATION_ACCESSORS_H_

/**
 * @file oskar_station_accessors.h
 */

#include <oskar_global.h>
#include <oskar_mem.h>
#include <oskar_element.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Data common to all station types. */

/**
 * @brief
 * Returns the unique identifier of the station within the telescope model.
 *
 * @details
 * Returns the unique identifier of the station within the telescope model.
 *
 * @param[in] model   Pointer to station model.
 *
 * @return The unique identifier.
 */
OSKAR_EXPORT
int oskar_station_unique_id(const oskar_Station* model);

/**
 * @brief
 * Returns the numerical precision of data stored in the station model.
 *
 * @details
 * Returns the numerical precision of data stored in the station model.
 *
 * @param[in] model   Pointer to station model.
 *
 * @return The data type (OSKAR_SINGLE or OSKAR_DOUBLE).
 */
OSKAR_EXPORT
int oskar_station_precision(const oskar_Station* model);

/**
 * @brief
 * Returns the memory location of data stored in the station model.
 *
 * @details
 * Returns the memory location of data stored in the station model.
 *
 * @param[in] model   Pointer to station model.
 *
 * @return The memory location (OSKAR_CPU or OSKAR_GPU).
 */
OSKAR_EXPORT
int oskar_station_mem_location(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_type(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_normalise_final_beam(const oskar_Station* model);

OSKAR_EXPORT
double oskar_station_lon_rad(const oskar_Station* model);

OSKAR_EXPORT
double oskar_station_lat_rad(const oskar_Station* model);

OSKAR_EXPORT
double oskar_station_alt_metres(const oskar_Station* model);

OSKAR_EXPORT
double oskar_station_polar_motion_x_rad(const oskar_Station* model);

OSKAR_EXPORT
double oskar_station_polar_motion_y_rad(const oskar_Station* model);

OSKAR_EXPORT
double oskar_station_beam_lon_rad(const oskar_Station* model);

OSKAR_EXPORT
double oskar_station_beam_lat_rad(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_beam_coord_type(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_noise_freq_hz(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_noise_freq_hz_const(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_noise_rms_jy(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_noise_rms_jy_const(const oskar_Station* model);

/* Data used only for Gaussian beam stations. */

OSKAR_EXPORT
double oskar_station_gaussian_beam_fwhm_rad(const oskar_Station* model);

OSKAR_EXPORT
double oskar_station_gaussian_beam_reference_freq_hz(const oskar_Station* model);

/* Data used only for aperture array stations. */

OSKAR_EXPORT
int oskar_station_identical_children(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_num_elements(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_num_element_types(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_normalise_array_pattern(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_enable_array_pattern(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_common_element_orientation(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_array_is_3d(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_apply_element_errors(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_apply_element_weight(const oskar_Station* model);

OSKAR_EXPORT
unsigned int oskar_station_seed_time_variable_errors(const oskar_Station* model);

OSKAR_EXPORT
double oskar_station_element_x_alpha_rad(const oskar_Station* model,
        int index);

OSKAR_EXPORT
double oskar_station_element_x_beta_rad(const oskar_Station* model,
        int index);

OSKAR_EXPORT
double oskar_station_element_x_gamma_rad(const oskar_Station* model,
        int index);

OSKAR_EXPORT
double oskar_station_element_y_alpha_rad(const oskar_Station* model,
        int index);

OSKAR_EXPORT
double oskar_station_element_y_beta_rad(const oskar_Station* model,
        int index);

OSKAR_EXPORT
double oskar_station_element_y_gamma_rad(const oskar_Station* model,
        int index);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_true_x_enu_metres(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_true_x_enu_metres_const(
        const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_true_y_enu_metres(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_true_y_enu_metres_const(
        const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_true_z_enu_metres(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_true_z_enu_metres_const(
        const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_measured_x_enu_metres(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_measured_x_enu_metres_const(
        const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_measured_y_enu_metres(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_measured_y_enu_metres_const(
        const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_measured_z_enu_metres(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_measured_z_enu_metres_const(
        const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_gain(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_gain_const(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_gain_error(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_gain_error_const(
        const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_phase_offset_rad(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_phase_offset_rad_const(
        const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_phase_error_rad(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_phase_error_rad_const(
        const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_weight(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_weight_const(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_types(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_types_const(const oskar_Station* model);

OSKAR_EXPORT
const int* oskar_station_element_types_cpu_const(const oskar_Station* model);

OSKAR_EXPORT
const char* oskar_station_element_mount_types_const(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_has_child(const oskar_Station* model);

OSKAR_EXPORT
oskar_Station* oskar_station_child(oskar_Station* model, int i);

OSKAR_EXPORT
const oskar_Station* oskar_station_child_const(const oskar_Station* model,
        int i);

OSKAR_EXPORT
int oskar_station_has_element(const oskar_Station* model);

OSKAR_EXPORT
oskar_Element* oskar_station_element(oskar_Station* model,
        int element_type_index);

OSKAR_EXPORT
const oskar_Element* oskar_station_element_const(const oskar_Station* model,
        int element_type_index);

OSKAR_EXPORT
int oskar_station_num_permitted_beams(const oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_permitted_beam_az_rad_const(
        const oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_permitted_beam_el_rad_const(
        const oskar_Station* model);


/* Setters. */

/**
 * @brief
 * Sets the unique station identifier.
 *
 * @details
 * Sets the unique station identifier.
 * This should be set only when the telescope model is being initialised.
 *
 * @param[in] model Pointer to station model.
 * @param[in] id    Identifier.
 */
OSKAR_EXPORT
void oskar_station_set_unique_id(oskar_Station* model, int id);

/**
 * @brief
 * Sets the station type (aperture array, Gaussian beam, etc).
 *
 * @details
 * Sets the station type (aperture array, Gaussian beam, etc).
 *
 * @param[in] model Pointer to station model.
 * @param[in] type  Enumerator describing the station type.
 */
OSKAR_EXPORT
void oskar_station_set_station_type(oskar_Station* model, int type);

/**
 * @brief
 * Sets the flag to specify whether the beam should be completely normalised
 * (default false).
 *
 * @details
 * Sets the flag to specify whether the station beam should be completely
 * normalised to a peak value of 1.0 (default false).
 *
 * @param[in] model Pointer to station model.
 * @param[in] value True or false.
 */
OSKAR_EXPORT
void oskar_station_set_normalise_final_beam(oskar_Station* model, int value);

/**
 * @brief
 * Sets the geographic coordinates of the station centre.
 *
 * @details
 * Sets the longitude, latitude and altitude of the station centre.
 *
 * @param[in] model          Pointer to station model.
 * @param[in] longitude_rad  East-positive longitude, in radians.
 * @param[in] latitude_rad   North-positive geodetic latitude in radians.
 * @param[in] altitude_m     Altitude above ellipsoid in metres.
 */
OSKAR_EXPORT
void oskar_station_set_position(oskar_Station* model,
        double longitude_rad, double latitude_rad, double altitude_m);

/**
 * @brief
 * Sets the polar motion components.
 *
 * @details
 * Sets the polar motion components for the station.
 * This function operates recursively to set polar motion components for all
 * child stations too.
 *
 * @param[in] model      Pointer to station model.
 * @param[in] pm_x_rad   Polar motion x-component, in radians.
 * @param[in] pm_y_rad   Polar motion y-component, in radians.
 */
OSKAR_EXPORT
void oskar_station_set_polar_motion(oskar_Station* model,
        double pm_x_rad, double pm_y_rad);

/**
 * @brief
 * Sets the coordinates of the phase centre.
 *
 * @details
 * Sets the longitude and latitude of the station phase centre on the sky.
 *
 * @param[in] model               Pointer to station model.
 * @param[in] beam_coord_type     Beam coordinate type (either
 *                                OSKAR_SPHERICAL_TYPE_EQUATORIAL or
 *                                OSKAR_SPHERICAL_TYPE_AZEL).
 * @param[in] beam_longitude_rad  Beam longitude in radians.
 * @param[in] beam_latitude_rad   Beam latitude in radians.
 */
OSKAR_EXPORT
void oskar_station_set_phase_centre(oskar_Station* model,
        int beam_coord_type, double beam_longitude_rad,
        double beam_latitude_rad);

/**
 * @brief
 * Sets the parameters of the Gaussian beam used for Gaussian beam stations.
 *
 * @details
 * Sets the parameters of the Gaussian beam used for Gaussian beam stations.
 *
 * @param[in] model       Pointer to station model.
 * @param[in] fwhm_rad    Full-width-half-maximum of the Gaussian beam, in radians.
 * @param[in] ref_freq_hz Reference frequency at which FWHM applies, in Hz.
 */
OSKAR_EXPORT
void oskar_station_set_gaussian_beam(oskar_Station* model,
        double fwhm_rad, double ref_freq_hz);

/**
 * @brief
 * Sets the flag to specify whether the beam should be normalised (default false).
 *
 * @details
 * Sets the flag to specify whether the station beam should be normalised
 * by dividing by the number of elements (default false).
 *
 * @param[in] model  Pointer to station model.
 * @param[in] value  True or false.
 */
OSKAR_EXPORT
void oskar_station_set_normalise_array_pattern(oskar_Station* model, int value);

/**
 * @brief
 * Sets the flag to specify whether the array pattern is enabled.
 *
 * @details
 * Sets the flag to specify whether the full station beam should be evaluated
 * by computing the array pattern (default true).
 *
 * @param[in] model  Pointer to station model.
 * @param[in] value  True or false.
 */
OSKAR_EXPORT
void oskar_station_set_enable_array_pattern(oskar_Station* model, int value);

/**
 * @brief
 * Sets the seed used to generate time-variable errors.
 *
 * @param[in] model  Pointer to station model.
 * @param[in] value  Seed value.
 */
OSKAR_EXPORT
void oskar_station_set_seed_time_variable_errors(oskar_Station* model,
        unsigned int value);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_STATION_ACCESSORS_H_ */
