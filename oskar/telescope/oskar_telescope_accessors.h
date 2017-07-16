/*
 * Copyright (c) 2013-2016, The University of Oxford
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

#ifndef OSKAR_TELESCOPE_ACCESSORS_H_
#define OSKAR_TELESCOPE_ACCESSORS_H_

/**
 * @file oskar_telescope_accessors.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>
#include <telescope/station/oskar_station.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Properties and metadata. */

/**
 * @brief
 * Returns the numerical precision of data stored in the telescope model.
 *
 * @details
 * Returns the numerical precision of data stored in the telescope model.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The data type (OSKAR_SINGLE or OSKAR_DOUBLE).
 */
OSKAR_EXPORT
int oskar_telescope_precision(const oskar_Telescope* model);

/**
 * @brief
 * Returns the memory location of data stored in the telescope model.
 *
 * @details
 * Returns the memory location of data stored in the telescope model.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The enumerated memory location.
 */
OSKAR_EXPORT
int oskar_telescope_mem_location(const oskar_Telescope* model);

/**
 * @brief
 * Returns the longitude of the telescope centre.
 *
 * @details
 * Returns the geodetic longitude of the interferometer centre in radians.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The longitude in radians.
 */
OSKAR_EXPORT
double oskar_telescope_lon_rad(const oskar_Telescope* model);

/**
 * @brief
 * Returns the latitude of the telescope centre.
 *
 * @details
 * Returns the geodetic latitude of the interferometer centre in radians.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The latitude in radians.
 */
OSKAR_EXPORT
double oskar_telescope_lat_rad(const oskar_Telescope* model);

/**
 * @brief
 * Returns the altitude of the telescope centre.
 *
 * @details
 * Returns the altitude of the interferometer centre above the ellipsoid,
 * in metres.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The altitude in metres.
 */
OSKAR_EXPORT
double oskar_telescope_alt_metres(const oskar_Telescope* model);

/**
 * @brief
 * Returns the x-component of polar motion.
 *
 * @details
 * Returns the x-component of polar motion.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The x-component of polar motion in radians.
 */
OSKAR_EXPORT
double oskar_telescope_polar_motion_x_rad(const oskar_Telescope* model);

/**
 * @brief
 * Returns the y-component of polar motion.
 *
 * @details
 * Returns the y-component of polar motion.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The y-component of polar motion in radians.
 */
OSKAR_EXPORT
double oskar_telescope_polar_motion_y_rad(const oskar_Telescope* model);

/**
 * @brief
 * Returns the enumerated phase centre coordinate type.
 *
 * @details
 * Returns the enumerated phase centre coordinate type.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The enumerated phase centre coordinate type.
 */
OSKAR_EXPORT
int oskar_telescope_phase_centre_coord_type(const oskar_Telescope* model);

/**
 * @brief
 * Returns the right ascension of the phase centre.
 *
 * @details
 * Returns the right ascension of the interferometer phase centre in radians.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The right ascension in radians.
 */
OSKAR_EXPORT
double oskar_telescope_phase_centre_ra_rad(const oskar_Telescope* model);

/**
 * @brief
 * Returns the declination of the phase centre.
 *
 * @details
 * Returns the declination of the interferometer phase centre in radians.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The declination in radians.
 */
OSKAR_EXPORT
double oskar_telescope_phase_centre_dec_rad(const oskar_Telescope* model);

/**
 * @brief
 * Returns the channel bandwidth in Hz.
 *
 * @details
 * Returns the channel bandwidth in Hz.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The channel bandwidth in Hz.
 */
OSKAR_EXPORT
double oskar_telescope_channel_bandwidth_hz(const oskar_Telescope* model);

/**
 * @brief
 * Returns the time averaging interval in seconds.
 *
 * @details
 * Returns the time averaging interval in seconds.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The time averaging interval in seconds.
 */
OSKAR_EXPORT
double oskar_telescope_time_average_sec(const oskar_Telescope* model);

/**
 * @brief
 * Returns the UV filter minimum bound.
 *
 * @details
 * Returns the UV filter minimum bound.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The UV filter minimum bound.
 */
OSKAR_EXPORT
double oskar_telescope_uv_filter_min(const oskar_Telescope* model);

/**
 * @brief
 * Returns the UV filter maximum bound.
 *
 * @details
 * Returns the UV filter maximum bound.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The UV filter maximum bound.
 */
OSKAR_EXPORT
double oskar_telescope_uv_filter_max(const oskar_Telescope* model);

/**
 * @brief
 * Returns the units of the UV filter values.
 *
 * @details
 * Returns the units of the UV filter values
 * (OSKAR_METRES or OSKAR_WAVELENGTHS).
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The units of the UV filter values.
 */
OSKAR_EXPORT
int oskar_telescope_uv_filter_units(const oskar_Telescope* model);

/**
 * @brief
 * Returns the polarisation mode of the telescope (full or scalar).
 *
 * @details
 * Returns the polarisation mode of the telescope
 * (OSKAR_POL_MODE_FULL or OSKAR_POL_MODE_SCALAR).
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The polarisation mode enumerator.
 */
OSKAR_EXPORT
int oskar_telescope_pol_mode(const oskar_Telescope* model);

/**
 * @brief
 * Returns the number of unique baselines in the telescope model.
 *
 * @details
 * Returns the number of unique baselines in the telescope model.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return The number of baselines.
 */
OSKAR_EXPORT
int oskar_telescope_num_baselines(const oskar_Telescope* model);

/**
 * @brief
 * Returns the number of interferometer stations in the telescope model.
 *
 * @details
 * Returns the number of interferometer stations in the telescope model.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return The number of stations.
 */
OSKAR_EXPORT
int oskar_telescope_num_stations(const oskar_Telescope* model);

/**
 * @brief
 * Returns a flag to specify whether all stations are identical.
 *
 * @details
 * Returns a flag to specify whether all stations are identical.
 *
 * Note that this flag is only valid after calling
 * oskar_telescope_analyse().
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return True if all stations are identical; false if not.
 */
OSKAR_EXPORT
int oskar_telescope_identical_stations(const oskar_Telescope* model);

/**
 * @brief
 * Returns the flag specifying whether station beam duplication is enabled.
 *
 * @details
 * Returns the flag specifying whether station beam duplication is enabled.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The boolean flag value.
 */
OSKAR_EXPORT
int oskar_telescope_allow_station_beam_duplication(
        const oskar_Telescope* model);

/**
 * @brief
 * Returns the flag specifying whether numerical element patterns are enabled.
 *
 * @details
 * Returns the flag specifying whether numerical element patterns are enabled.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The boolean flag value.
 */
OSKAR_EXPORT
int oskar_telescope_enable_numerical_patterns(const oskar_Telescope* model);

/**
 * @brief
 * Returns the maximum number of elements in a station.
 *
 * @details
 * Returns the maximum number of elements in a station.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The maximum number of elements in a station.
 */
OSKAR_EXPORT
int oskar_telescope_max_station_size(const oskar_Telescope* model);

/**
 * @brief
 * Returns the maximum beamforming hierarchy depth.
 *
 * @details
 * Returns the maximum beamforming hierarchy depth.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The maximum beamforming hierarchy depth.
 */
OSKAR_EXPORT
int oskar_telescope_max_station_depth(const oskar_Telescope* model);


/* Station models. */

/**
 * @brief
 * Returns a handle to a station model at the given index.
 *
 * @details
 * Returns a handle to a station model at the given index.
 *
 * @param[in] model Pointer to telescope model.
 * @param[in] i     The station model index.
 *
 * @return A handle to a station model.
 */
OSKAR_EXPORT
oskar_Station* oskar_telescope_station(oskar_Telescope* model, int i);

/**
 * @brief
 * Returns a constant handle to a station model at the given index.
 *
 * @details
 * Returns a constant handle to a station model at the given index.
 *
 * @param[in] model Pointer to telescope model.
 * @param[in] i     The station model index.
 *
 * @return A constant handle to a station model.
 */
OSKAR_EXPORT
const oskar_Station* oskar_telescope_station_const(
        const oskar_Telescope* model, int i);


/* Coordinate arrays. */

/**
 * @brief
 * Returns a handle to the measured station x positions.
 *
 * @details
 * Returns a handle to the measured station x positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A handle to the measured station x positions.
 */
OSKAR_EXPORT
oskar_Mem* oskar_telescope_station_measured_x_offset_ecef_metres(
        oskar_Telescope* model);

/**
 * @brief
 * Returns a constant handle to the measured station x positions.
 *
 * @details
 * Returns a constant handle to the measured station x positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A constant handle to the measured station x positions.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_telescope_station_measured_x_offset_ecef_metres_const(
        const oskar_Telescope* model);

/**
 * @brief
 * Returns a handle to the measured station y positions.
 *
 * @details
 * Returns a handle to the measured station y positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A handle to the measured station y positions.
 */
OSKAR_EXPORT
oskar_Mem* oskar_telescope_station_measured_y_offset_ecef_metres(
        oskar_Telescope* model);

/**
 * @brief
 * Returns a constant handle to the measured station y positions.
 *
 * @details
 * Returns a constant handle to the measured station y positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A constant handle to the measured station y positions.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_telescope_station_measured_y_offset_ecef_metres_const(
        const oskar_Telescope* model);

/**
 * @brief
 * Returns a handle to the measured station z positions.
 *
 * @details
 * Returns a handle to the measured station z positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A handle to the measured station z positions.
 */
OSKAR_EXPORT
oskar_Mem* oskar_telescope_station_measured_z_offset_ecef_metres(
        oskar_Telescope* model);

/**
 * @brief
 * Returns a constant handle to the measured station z positions.
 *
 * @details
 * Returns a constant handle to the measured station z positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A constant handle to the measured station z positions.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_telescope_station_measured_z_offset_ecef_metres_const(
        const oskar_Telescope* model);

/**
 * @brief
 * Returns a handle to the measured horizon plane station x positions.
 *
 * @details
 * Returns a handle to the measured horizon plane station x positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A handle to the measured horizon plane station x positions.
 */
OSKAR_EXPORT
oskar_Mem* oskar_telescope_station_measured_x_enu_metres(oskar_Telescope* model);

/**
 * @brief
 * Returns a constant handle to the measured horizon plane station x positions.
 *
 * @details
 * Returns a constant handle to the measured horizon plane station x positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A constant handle to the measured horizon plane station x positions.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_telescope_station_measured_x_enu_metres_const(
        const oskar_Telescope* model);

/**
 * @brief
 * Returns a handle to the measured horizon plane station y positions.
 *
 * @details
 * Returns a handle to the measured horizon plane station y positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A handle to the measured horizon plane station y positions.
 */
OSKAR_EXPORT
oskar_Mem* oskar_telescope_station_measured_y_enu_metres(oskar_Telescope* model);

/**
 * @brief
 * Returns a constant handle to the measured horizon plane station y positions.
 *
 * @details
 * Returns a constant handle to the measured horizon plane station y positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A constant handle to the measured horizon plane station y positions.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_telescope_station_measured_y_enu_metres_const(
        const oskar_Telescope* model);

/**
 * @brief
 * Returns a handle to the measured horizon plane station z positions.
 *
 * @details
 * Returns a handle to the measured horizon plane station z positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A handle to the measured horizon plane station z positions.
 */
OSKAR_EXPORT
oskar_Mem* oskar_telescope_station_measured_z_enu_metres(oskar_Telescope* model);

/**
 * @brief
 * Returns a constant handle to the measured horizon plane station z positions.
 *
 * @details
 * Returns a constant handle to the measured horizon plane station z positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A constant handle to the measured horizon plane station z positions.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_telescope_station_measured_z_enu_metres_const(
        const oskar_Telescope* model);

/**
 * @brief
 * Returns a handle to the true station x positions.
 *
 * @details
 * Returns a handle to the true station x positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A handle to the true station x positions.
 */
OSKAR_EXPORT
oskar_Mem* oskar_telescope_station_true_x_offset_ecef_metres(
        oskar_Telescope* model);

/**
 * @brief
 * Returns a constant handle to the true station x positions.
 *
 * @details
 * Returns a constant handle to the true station x positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A constant handle to the true station x positions.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_telescope_station_true_x_offset_ecef_metres_const(
        const oskar_Telescope* model);

/**
 * @brief
 * Returns a handle to the true station y positions.
 *
 * @details
 * Returns a handle to the true station y positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A handle to the true station y positions.
 */
OSKAR_EXPORT
oskar_Mem* oskar_telescope_station_true_y_offset_ecef_metres(
        oskar_Telescope* model);

/**
 * @brief
 * Returns a constant handle to the true station y positions.
 *
 * @details
 * Returns a constant handle to the true station y positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A constant handle to the true station y positions.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_telescope_station_true_y_offset_ecef_metres_const(
        const oskar_Telescope* model);

/**
 * @brief
 * Returns a handle to the true station z positions.
 *
 * @details
 * Returns a handle to the true station z positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A handle to the true station z positions.
 */
OSKAR_EXPORT
oskar_Mem* oskar_telescope_station_true_z_offset_ecef_metres(
        oskar_Telescope* model);

/**
 * @brief
 * Returns a constant handle to the true station z positions.
 *
 * @details
 * Returns a constant handle to the true station z positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A constant handle to the true station z positions.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_telescope_station_true_z_offset_ecef_metres_const(
        const oskar_Telescope* model);

/**
 * @brief
 * Returns a handle to the true horizon plane station x positions.
 *
 * @details
 * Returns a handle to the true horizon plane station x positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A handle to the true horizon plane station x positions.
 */
OSKAR_EXPORT
oskar_Mem* oskar_telescope_station_true_x_enu_metres(oskar_Telescope* model);

/**
 * @brief
 * Returns a constant handle to the true horizon plane station x positions.
 *
 * @details
 * Returns a constant handle to the true horizon plane station x positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A constant handle to the true horizon plane station x positions.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_telescope_station_true_x_enu_metres_const(
        const oskar_Telescope* model);

/**
 * @brief
 * Returns a handle to the true horizon plane station y positions.
 *
 * @details
 * Returns a handle to the true horizon plane station y positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A handle to the true horizon plane station y positions.
 */
OSKAR_EXPORT
oskar_Mem* oskar_telescope_station_true_y_enu_metres(oskar_Telescope* model);

/**
 * @brief
 * Returns a constant handle to the true horizon plane station y positions.
 *
 * @details
 * Returns a constant handle to the true horizon plane station y positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A constant handle to the true horizon plane station y positions.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_telescope_station_true_y_enu_metres_const(
        const oskar_Telescope* model);

/**
 * @brief
 * Returns a handle to the true horizon plane station z positions.
 *
 * @details
 * Returns a handle to the true horizon plane station z positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A handle to the true horizon plane station z positions.
 */
OSKAR_EXPORT
oskar_Mem* oskar_telescope_station_true_z_enu_metres(oskar_Telescope* model);

/**
 * @brief
 * Returns a constant handle to the true horizon plane station z positions.
 *
 * @details
 * Returns a constant handle to the true horizon plane station z positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A constant handle to the true horizon plane station z positions.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_telescope_station_true_z_enu_metres_const(
        const oskar_Telescope* model);

/**
 * @brief
 * Returns the flag specifying whether thermal noise is enabled.
 *
 * @details
 * Returns the flag specifying whether thermal noise is enabled.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return Flag specifying whether thermal noise is enabled.
 */
OSKAR_EXPORT
int oskar_telescope_noise_enabled(const oskar_Telescope* model);

/**
 * @brief
 * Returns the random generator seed.
 *
 * @details
 * Returns the random generator seed.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return The random generator seed.
 */
OSKAR_EXPORT
unsigned int oskar_telescope_noise_seed(const oskar_Telescope* model);


/* Setters. */

/**
 * @brief
 * Sets the flag to specify whether station beam duplication is enabled.
 *
 * @details
 * Sets the flag to specify whether station beam duplication is enabled.
 *
 * @param[in] model    Pointer to telescope model.
 * @param[in] value    If true, stations will share common source positions.
 */
OSKAR_EXPORT
void oskar_telescope_set_allow_station_beam_duplication(oskar_Telescope* model,
        int value);

/**
 * @brief
 * Sets whether thermal noise is enabled.
 *
 * @details
 * Sets whether thermal noise is enabled.
 *
 * @param[in] model            Pointer to telescope model.
 * @param[in] value            If true, enable thermal noise.
 * @param[in] seed             Random generator seed.
 */
OSKAR_EXPORT
void oskar_telescope_set_enable_noise(oskar_Telescope* model,
        int value, unsigned int seed);

/**
 * @brief
 * Sets the flag to specify whether numerical element patterns are enabled.
 *
 * @details
 * Sets the flag to specify whether numerical element patterns are enabled.
 *
 * @param[in] model    Pointer to telescope model.
 * @param[in] value    If true, numerical element patterns will be enabled.
 */
OSKAR_EXPORT
void oskar_telescope_set_enable_numerical_patterns(oskar_Telescope* model,
        int value);

/**
 * @brief
 * Sets the Gaussian station beam parameters.
 *
 * @details
 * Sets the Gaussian station beam parameters.
 * These are only used if the station type is "Gaussian beam"
 *
 * @param[in] model       Pointer to telescope model.
 * @param[in] fwhm_deg    The Gaussian FWHM value of the beam, in degrees.
 * @param[in] ref_freq_hz Reference frequency at which the FWHM applies, in Hz.
 */
OSKAR_EXPORT
void oskar_telescope_set_gaussian_station_beam_width(oskar_Telescope* model,
        double fwhm_deg, double ref_freq_hz);

/**
 * @brief
 * Sets the frequencies for which thermal noise is defined.
 *
 * @details
 * Sets the frequencies for which thermal noise is defined.
 *
 * @param[in] model            Pointer to telescope model.
 * @param[in] filename         Text file to load.
 * @param[in,out] status       Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_set_noise_freq_file(oskar_Telescope* model,
        const char* filename, int* status);

/**
 * @brief
 * Sets the frequencies for which thermal noise is defined.
 *
 * @details
 * Sets the frequencies for which thermal noise is defined.
 *
 * @param[in] model            Pointer to telescope model.
 * @param[in] start_hz         Frequency of the first channel, in Hz.
 * @param[in] inc_hz           Frequency increment, in Hz.
 * @param[in] num_channels     Number of frequency channels.
 * @param[in,out] status       Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_set_noise_freq(oskar_Telescope* model,
        double start_hz, double inc_hz, int num_channels, int* status);

/**
 * @brief
 * Sets the thermal noise RMS values from a file.
 *
 * @details
 * Sets the thermal noise RMS values from a file.
 *
 * @param[in] model            Pointer to telescope model.
 * @param[in] filename         Text file to load.
 * @param[in,out] status       Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_set_noise_rms_file(oskar_Telescope* model,
        const char* filename, int* status);

/**
 * @brief
 * Sets the thermal noise RMS values from a range.
 *
 * @details
 * Sets the thermal noise RMS values from a range.
 *
 * Note that this can only be called after the noise frequencies have been
 * defined.
 *
 * @param[in] model            Pointer to telescope model.
 * @param[in] start            RMS value in the first channel.
 * @param[in] end              RMS value in the last channel.
 * @param[in,out] status       Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_set_noise_rms(oskar_Telescope* model,
        double start, double end, int* status);

/**
 * @brief
 * Sets the geographic coordinates of the telescope centre.
 *
 * @details
 * Sets the longitude, latitude and altitude of the interferometer centre.
 *
 * @param[in] model            Pointer to telescope model.
 * @param[in] longitude_rad    East-positive longitude, in radians.
 * @param[in] latitude_rad     North-positive geodetic latitude in radians.
 * @param[in] altitude_metres  Altitude above ellipsoid in metres.
 */
OSKAR_EXPORT
void oskar_telescope_set_position(oskar_Telescope* model,
        double longitude_rad, double latitude_rad, double altitude_metres);

/**
 * @brief
 * Sets the polar motion components.
 *
 * @details
 * Sets the polar motion components into the telescope model.
 * This function recursively sets polar motion components for all existing
 * stations too.
 *
 * @param[in] model      Pointer to station model.
 * @param[in] pm_x_rad   Polar motion x-component, in radians.
 * @param[in] pm_y_rad   Polar motion y-component, in radians.
 */
OSKAR_EXPORT
void oskar_telescope_set_polar_motion(oskar_Telescope* model,
        double pm_x_rad, double pm_y_rad);

/**
 * @brief
 * Sets the coordinates of the phase centre.
 *
 * @details
 * Sets the right ascension and declination of the interferometer phase centre.
 *
 * @param[in] model       Pointer to telescope model.
 * @param[in] coord_type  Coordinate type (ICRS or CIRS).
 * @param[in] ra_rad      Right ascension in radians.
 * @param[in] dec_rad     Declination in radians.
 */
OSKAR_EXPORT
void oskar_telescope_set_phase_centre(oskar_Telescope* model,
        int coord_type, double ra_rad, double dec_rad);

/**
 * @brief
 * Sets the channel bandwidth, used for bandwidth smearing.
 *
 * @details
 * Sets the channel bandwidth, used for bandwidth smearing.
 *
 * @param[in] model            Pointer to telescope model.
 * @param[in] bandwidth_hz     Channel bandwidth, in Hz.
 */
OSKAR_EXPORT
void oskar_telescope_set_channel_bandwidth(oskar_Telescope* model,
        double bandwidth_hz);

/**
 * @brief
 * Sets the time average interval, used for time-average smearing.
 *
 * @details
 * Sets the time average interval, used for time-average smearing.
 *
 * @param[in] model            Pointer to telescope model.
 * @param[in] time_average_sec Time averaging interval, in seconds.
 */
OSKAR_EXPORT
void oskar_telescope_set_time_average(oskar_Telescope* model,
        double time_average_sec);

/**
 * @brief
 * Sets unique station IDs in a telescope model.
 *
 * @details
 * This function sets unique station IDs in a telescope model,
 * recursively if necessary.
 *
 * @param[in] model            Pointer to telescope model.
 */
OSKAR_EXPORT
void oskar_telescope_set_station_ids(oskar_Telescope* model);

/**
 * @brief
 * Sets the type of stations within the telescope model.
 *
 * @details
 * Sets the type of stations within the telescope model,
 * recursively if necessary.
 *
 * Only the first letter of the type string is checked.
 *
 * @param[in] model            Pointer to telescope model.
 * @param[in] type             Station type, either "Array", "Gaussian"
 *                             or "Isotropic".
 * @param[in,out] status       Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_set_station_type(oskar_Telescope* model, const char* type,
        int* status);

/**
 * @brief
 * Sets the baseline UV range to evaluate.
 *
 * @details
 * Sets the baseline UV range to evaluate.
 * Baselines with lengths outside this range will not be evaluated.
 *
 * @param[in] model            Pointer to telescope model.
 * @param[in] uv_filter_min    Minimum value for UV filter.
 * @param[in] uv_filter_max    Maximum value for UV filter.
 * @param[in] units            Units of UV filter ("Metres" or "Wavelengths").
 */
OSKAR_EXPORT
void oskar_telescope_set_uv_filter(oskar_Telescope* model,
        double uv_filter_min, double uv_filter_max, const char* units,
        int* status);

/**
 * @brief
 * Sets the polarisation mode of the telescope.
 *
 * @details
 * Sets the polarisation mode of the telescope.
 *
 * @param[in] model       Pointer to telescope model.
 * @param[in] mode        Mode string ("Full" or "Scalar").
 * @param[in,out] status  Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_set_pol_mode(oskar_Telescope* model, const char* mode,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_TELESCOPE_ACCESSORS_H_ */
