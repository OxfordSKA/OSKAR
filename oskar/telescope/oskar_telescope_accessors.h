/*
 * Copyright (c) 2013-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_TELESCOPE_ACCESSORS_H_
#define OSKAR_TELESCOPE_ACCESSORS_H_

/**
 * @file oskar_telescope_accessors.h
 */

#include <oskar_global.h>
#include <gains/oskar_gains.h>
#include <harp/oskar_harp.h>
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
 * Returns the longitude of the phase centre.
 *
 * @details
 * Returns the longitude of the interferometer phase centre in radians.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The longitude in radians.
 */
OSKAR_EXPORT
double oskar_telescope_phase_centre_longitude_rad(const oskar_Telescope* model);

/**
 * @brief
 * Returns the latitude of the phase centre.
 *
 * @details
 * Returns the latitude of the interferometer phase centre in radians.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The latitude in radians.
 */
OSKAR_EXPORT
double oskar_telescope_phase_centre_latitude_rad(const oskar_Telescope* model);

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
 * Returns the TEC screen height, in km.
 *
 * @details
 * Returns the TEC screen height, in km.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The TEC screen height, in km.
 */
OSKAR_EXPORT
double oskar_telescope_tec_screen_height_km(const oskar_Telescope* model);

/**
 * @brief
 * Returns the TEC screen pixel size, in metres.
 *
 * @details
 * Returns the TEC screen pixel size, in metres.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The TEC screen pixel size, in metres.
 */
OSKAR_EXPORT
double oskar_telescope_tec_screen_pixel_size_m(const oskar_Telescope* model);

/**
 * @brief
 * Returns the TEC screen time interval, in seconds.
 *
 * @details
 * Returns the TEC screen time interval, in seconds.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return The TEC screen time interval, in seconds.
 */
OSKAR_EXPORT
double oskar_telescope_tec_screen_time_interval_sec(
        const oskar_Telescope* model);

/**
 * @brief
 * Returns the flag to specify whether isoplanatic screens should be used.
 *
 * @details
 * Returns the flag to specify whether isoplanatic screens should be used.
 *
 * @param[in] model   Pointer to telescope model.
 *
 * @return If true, screens should be treated as isoplanatic.
 */
OSKAR_EXPORT
int oskar_telescope_isoplanatic_screen(const oskar_Telescope* model);

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
 * Returns the number of station models in the telescope model.
 *
 * @details
 * Returns the number of station models in the telescope model.
 * This is not necessarily the same as the number of stations,
 * if beam duplication is allowed.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return The number of stations.
 */
OSKAR_EXPORT
int oskar_telescope_num_station_models(const oskar_Telescope* model);

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
 * Returns the flag specifying whether an ionospheric phase screen is enabled.
 *
 * @details
 * Returns the flag specifying whether an ionospheric phase screen is enabled.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return Flag specifying whether an ionospheric phase screen is enabled.
 */
OSKAR_EXPORT
char oskar_telescope_ionosphere_screen_type(const oskar_Telescope* model);

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


/* Arrays. */

/**
 * @brief
 * Returns a handle to the station type mapping array.
 *
 * @details
 * Returns a handle to the station type mapping array.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A handle to the station type mapping array.
 */
OSKAR_EXPORT
oskar_Mem* oskar_telescope_station_type_map(oskar_Telescope* model);

/**
 * @brief
 * Returns a constant handle to the station type mapping array.
 *
 * @details
 * Returns a constant handle to the station type mapping array.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A constant handle to the station type mapping array.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_telescope_station_type_map_const(
        const oskar_Telescope* model);

/**
 * @brief
 * Returns a handle to the measured station positions.
 *
 * @details
 * Returns a handle to the measured station positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A handle to the measured station positions.
 */
OSKAR_EXPORT
oskar_Mem* oskar_telescope_station_measured_offset_ecef_metres(
        oskar_Telescope* model, int dim);

/**
 * @brief
 * Returns a constant handle to the measured station positions.
 *
 * @details
 * Returns a constant handle to the measured station positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A constant handle to the measured station positions.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_telescope_station_measured_offset_ecef_metres_const(
        const oskar_Telescope* model, int dim);

/**
 * @brief
 * Returns a handle to the measured horizon plane station positions.
 *
 * @details
 * Returns a handle to the measured horizon plane station positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A handle to the measured horizon plane station positions.
 */
OSKAR_EXPORT
oskar_Mem* oskar_telescope_station_measured_enu_metres(
        oskar_Telescope* model, int dim);

/**
 * @brief
 * Returns a constant handle to the measured horizon plane station positions.
 *
 * @details
 * Returns a constant handle to the measured horizon plane station positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A constant handle to the measured horizon plane station positions.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_telescope_station_measured_enu_metres_const(
        const oskar_Telescope* model, int dim);

/**
 * @brief
 * Returns a handle to the true station positions.
 *
 * @details
 * Returns a handle to the true station positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A handle to the true station positions.
 */
OSKAR_EXPORT
oskar_Mem* oskar_telescope_station_true_offset_ecef_metres(
        oskar_Telescope* model, int dim);

/**
 * @brief
 * Returns a constant handle to the true station positions.
 *
 * @details
 * Returns a constant handle to the true station positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A constant handle to the true station positions.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_telescope_station_true_offset_ecef_metres_const(
        const oskar_Telescope* model, int dim);

/**
 * @brief
 * Returns a handle to the true horizon plane station positions.
 *
 * @details
 * Returns a handle to the true horizon plane station positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A handle to the true horizon plane station positions.
 */
OSKAR_EXPORT
oskar_Mem* oskar_telescope_station_true_enu_metres(
        oskar_Telescope* model, int dim);

/**
 * @brief
 * Returns a constant handle to the true horizon plane station positions.
 *
 * @details
 * Returns a constant handle to the true horizon plane station positions.
 *
 * @param[in] model Pointer to telescope model.
 *
 * @return A constant handle to the true horizon plane station positions.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_telescope_station_true_enu_metres_const(
        const oskar_Telescope* model, int dim);

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

/**
 * @brief
 * Returns the path to an externally-generated TEC screen.
 *
 * @details
 * Returns the path to an externally-generated TEC screen.
 *
 * @param[in] model    Pointer to telescope model.
 *
 * @return The path to the TEC screen.
 */
OSKAR_EXPORT
const char* oskar_telescope_tec_screen_path(const oskar_Telescope* model);

/**
 * @brief
 * Returns the gain model.
 *
 * @details
 * Returns the gain model.
 *
 * @param[in] model    Pointer to telescope model.
 *
 * @return The gain model.
 */
OSKAR_EXPORT
oskar_Gains* oskar_telescope_gains(oskar_Telescope* model);

/**
 * @brief
 * Returns the gain model.
 *
 * @details
 * Returns the gain model.
 *
 * @param[in] model    Pointer to telescope model.
 *
 * @return The gain model.
 */
OSKAR_EXPORT
const oskar_Gains* oskar_telescope_gains_const(const oskar_Telescope* model);

/**
 * @brief
 * Returns the HARP data model.
 *
 * @details
 * Returns the HARP data model.
 *
 * @param[in] model    Pointer to telescope model.
 * @param[in] freq_hz  The current observing frequency, in Hz.
 *
 * @return The HARP data model.
 */
OSKAR_EXPORT
oskar_Harp* oskar_telescope_harp_data(oskar_Telescope* model,
        double freq_hz);

/**
 * @brief
 * Returns the HARP data model.
 *
 * @details
 * Returns the HARP data model.
 *
 * @param[in] model    Pointer to telescope model.
 * @param[in] freq_hz  The current observing frequency, in Hz.
 *
 * @return The HARP data model.
 */
OSKAR_EXPORT
const oskar_Harp* oskar_telescope_harp_data_const(const oskar_Telescope* model,
        double freq_hz);


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
 * Sets the ionosphere screen type.
 *
 * @details
 * Sets the ionosphere screen type.
 *
 * @param[in] model    Pointer to telescope model.
 * @param[in] type     Type of screen to use (currently "None" or "External").
 */
OSKAR_EXPORT
void oskar_telescope_set_ionosphere_screen_type(oskar_Telescope* model,
        const char* type);

/**
 * @brief
 * Sets the option to treat phase screens as isoplanatic.
 *
 * @details
 * Sets the option to treat phase screens as isoplanatic.
 *
 * @param[in] model    Pointer to telescope model.
 * @param[in] flag     If true, treat phase screens as isoplanatic.
 */
OSKAR_EXPORT
void oskar_telescope_set_isoplanatic_screen(oskar_Telescope* model, int flag);

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
 * Sets the coordinates of the phase centre.
 *
 * @details
 * Sets the coordinates of the interferometer phase centre.
 *
 * @param[in] model         Pointer to telescope model.
 * @param[in] coord_type    Coordinate type (OSKAR_SPHERICAL_TYPE enumerator).
 * @param[in] longitude_rad Longitude in radians.
 * @param[in] latitude_rad  Latitude in radians.
 */
OSKAR_EXPORT
void oskar_telescope_set_phase_centre(oskar_Telescope* model,
        int coord_type, double longitude_rad, double latitude_rad);

/**
 * @brief
 * Sets the path to an externally-generated phase screen.
 *
 * @details
 * Sets the path to an externally-generated phase screen.
 *
 * @param[in] model    Pointer to telescope model.
 * @param[in] path     Path to FITS file to use as a phase screen.
 */
OSKAR_EXPORT
void oskar_telescope_set_tec_screen_path(oskar_Telescope* model,
        const char* path);

/**
 * @brief
 * Sets the TEC screen height.
 *
 * @details
 * Sets the TEC screen height.
 *
 * @param[in] model     Pointer to telescope model.
 * @param[in] height_km TEC screen height, in km.
 */
OSKAR_EXPORT
void oskar_telescope_set_tec_screen_height(oskar_Telescope* model,
        double height_km);

/**
 * @brief
 * Sets the TEC screen pixel size, in metres.
 *
 * @details
 * Sets the TEC screen pixel size, in metres.
 *
 * @param[in] model     Pointer to telescope model.
 * @param[in] pixel_size_m TEC screen pixel size, in metres.
 */
OSKAR_EXPORT
void oskar_telescope_set_tec_screen_pixel_size(oskar_Telescope* model,
        double pixel_size_m);

/**
 * @brief
 * Sets the TEC screen time interval, in seconds.
 *
 * @details
 * Sets the TEC screen time interval, in seconds.
 *
 * @param[in] model     Pointer to telescope model.
 * @param[in] time_interval_sec TEC screen time interval, in seconds.
 */
OSKAR_EXPORT
void oskar_telescope_set_tec_screen_time_interval(oskar_Telescope* model,
        double time_interval_sec);

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
 * Sets or clears the station type mapping.
 *
 * @details
 * Sets or clears the station type mapping.
 *
 * @param[in] model       Pointer to telescope model.
 * @param[in] value       If true, set the type map to specify all stations
 *                        are unique; otherwise, set all stations to be
 *                        the same.
 * @param[in,out] status  Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_set_unique_stations(oskar_Telescope* model,
        int value, int* status);

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

#ifdef __cplusplus
}
#endif

#endif /* include guard */
