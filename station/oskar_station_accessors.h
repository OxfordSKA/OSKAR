/*
 * Copyright (c) 2013, The University of Oxford
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
#include <oskar_ElementModel.h>
#include <oskar_SystemNoiseModel.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Data common to all station types. */

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
int oskar_station_type(const oskar_Station* model);

/**
 * @brief
 * Returns the memory location of data stored in the station model.
 *
 * @details
 * Returns the memory location of data stored in the station model.
 *
 * @param[in] model   Pointer to station model.
 *
 * @return The memory location (OSKAR_LOCATION_CPU or OSKAR_LOCATION_GPU).
 */
OSKAR_EXPORT
int oskar_station_location(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_station_type(const oskar_Station* model);

OSKAR_EXPORT
double oskar_station_longitude_rad(const oskar_Station* model);

OSKAR_EXPORT
double oskar_station_latitude_rad(const oskar_Station* model);

OSKAR_EXPORT
double oskar_station_altitude_m(const oskar_Station* model);

OSKAR_EXPORT
double oskar_station_beam_longitude_rad(const oskar_Station* model);

OSKAR_EXPORT
double oskar_station_beam_latitude_rad(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_beam_coord_type(const oskar_Station* model);

OSKAR_EXPORT
oskar_SystemNoiseModel* oskar_station_system_noise_model(oskar_Station* model);

OSKAR_EXPORT
const oskar_SystemNoiseModel* oskar_station_system_noise_model_const(const oskar_Station* model);


/* Data used only for Gaussian beam stations. */

OSKAR_EXPORT
double oskar_station_gaussian_beam_fwhm_rad(const oskar_Station* model);


/* Data used only for aperture array stations. */

OSKAR_EXPORT
int oskar_station_num_elements(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_num_element_types(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_use_polarised_elements(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_normalise_beam(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_enable_array_pattern(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_single_element_model(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_array_is_3d(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_apply_element_errors(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_apply_element_weight(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_element_coord_units(const oskar_Station* model);

OSKAR_EXPORT
double oskar_station_element_orientation_x_rad(const oskar_Station* model);

OSKAR_EXPORT
double oskar_station_element_orientation_y_rad(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_x_signal(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_x_signal_const(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_y_signal(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_y_signal_const(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_z_signal(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_z_signal_const(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_x_weights(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_x_weights_const(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_y_weights(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_y_weights_const(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_z_weights(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_z_weights_const(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_gain(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_gain_const(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_gain_error(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_gain_error_const(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_phase_offset(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_phase_offset_const(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_phase_error(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_phase_error_const(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_weight(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_weight_const(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_cos_orientation_x(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_cos_orientation_x_const(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_sin_orientation_x(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_sin_orientation_x_const(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_cos_orientation_y(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_cos_orientation_y_const(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_sin_orientation_y(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_sin_orientation_y_const(const oskar_Station* model);

OSKAR_EXPORT
oskar_Mem* oskar_station_element_type(oskar_Station* model);

OSKAR_EXPORT
const oskar_Mem* oskar_station_element_type_const(const oskar_Station* model);

OSKAR_EXPORT
int oskar_station_has_child(const oskar_Station* model);

OSKAR_EXPORT
oskar_Station* oskar_station_child(oskar_Station* model, int i);

OSKAR_EXPORT
const oskar_Station* oskar_station_child_const(const oskar_Station* model, int i);

OSKAR_EXPORT
int oskar_station_has_element(const oskar_Station* model);

OSKAR_EXPORT
oskar_ElementModel* oskar_station_element(oskar_Station* model, int i);

OSKAR_EXPORT
const oskar_ElementModel* oskar_station_element_const(const oskar_Station* model, int i);


/* Setters. */

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
 * Sets the coordinates of the phase centre.
 *
 * @details
 * Sets the longitude and latitude of the station phase centre on the sky.
 *
 * @param[in] model               Pointer to station model.
 * @param[in] beam_coord_type     Beam coordinate type (either
 *                                OSKAR_SPHERICAL_TYPE_EQUATORIAL or
 *                                OSKAR_SPHERICAL_TYPE_HORIZONTAL).
 * @param[in] beam_longitude_rad  Beam longitude in radians.
 * @param[in] beam_latitude_rad   Beam latitude in radians.
 */
OSKAR_EXPORT
void oskar_station_set_phase_centre(oskar_Station* model,
        int beam_coord_type, double beam_longitude_rad,
        double beam_latitude_rad);

/**
 * @brief
 * Sets the FWHM value of the Gaussian beam used for Gaussian beam stations.
 *
 * @details
 * Sets the FWHM value of the Gaussian beam used for Gaussian beam stations.
 *
 * @param[in] model Pointer to station model.
 * @param[in] value Full-width-half-maximum of the Gaussian beam, in radians.
 */
OSKAR_EXPORT
void oskar_station_set_gaussian_beam_fwhm_rad(oskar_Station* model,
        double value);

/**
 * @brief
 * Sets the flag to specify whether elements are polarised (default true).
 *
 * @details
 * Sets the flag to specify whether elements are polarised (default true).
 *
 * @param[in] model  Pointer to station model.
 * @param[in] value  True or false.
 */
OSKAR_EXPORT
void oskar_station_set_use_polarised_elements(oskar_Station* model, int value);

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
void oskar_station_set_normalise_beam(oskar_Station* model, int value);

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

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_STATION_ACCESSORS_H_ */
