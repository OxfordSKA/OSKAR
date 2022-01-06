/*
 * Copyright (c) 2011-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MS_ACCESSORS_H_
#define OSKAR_MS_ACCESSORS_H_

/**
 * @file oskar_ms_accessors.h
 */

#include <ms/oskar_ms_macros.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Returns the size in bytes of an element in a column of the Measurement Set.
 *
 * @details
 * Returns the size in bytes of an element in a column of the Measurement Set.
 *
 * @param[in] column     Column name.
 */
OSKAR_MS_EXPORT
size_t oskar_ms_column_element_size(const oskar_MeasurementSet* p,
        const char* column);

/**
 * @brief
 * Returns the data type of an element in a column of the Measurement Set.
 *
 * @details
 * Returns the data type of an element in a column of the Measurement Set.
 *
 * This is one of the values from the OSKAR_MS_TYPE enumerator.
 *
 * @param[in] column     Column name.
 */
OSKAR_MS_EXPORT
int oskar_ms_column_element_type(const oskar_MeasurementSet* p,
        const char* column);

/**
 * @brief
 * Returns the shape of an element in a column of the Measurement Set.
 *
 * @details
 * Returns the shape of an element in a column of the Measurement Set.
 *
 * Note that the returned array must be freed by the caller using free().
 *
 * @param[in] column     Column name.
 * @param[out] ndim      Number of dimensions.
 *
 * @return Array containing the size of each dimension.
 * Must be freed by the caller using free().
 */
OSKAR_MS_EXPORT
size_t* oskar_ms_column_shape(const oskar_MeasurementSet* p, const char* column,
        size_t* ndim);

/**
 * @brief
 * Ensures the specified number of rows exist in the Measurement Set.
 *
 * @details
 * Ensures the specified number of rows exist in the Measurement Set,
 * adding extra ones if necessary.
 *
 * @param[in] num    Total number of rows in the Measurement Set.
 */
OSKAR_MS_EXPORT
void oskar_ms_ensure_num_rows(oskar_MeasurementSet* p, unsigned int num);

/**
 * @brief
 * Returns the channel separation in the Measurement Set.
 *
 * @details
 * Returns the channel separation in the Measurement Set.
 */
OSKAR_MS_EXPORT
double oskar_ms_freq_inc_hz(const oskar_MeasurementSet* p);

/**
 * @brief
 * Returns the reference frequency in the Measurement Set.
 *
 * @details
 * Returns the reference frequency in the Measurement Set.
 */
OSKAR_MS_EXPORT
double oskar_ms_freq_start_hz(const oskar_MeasurementSet* p);

/**
 * @brief
 * Returns the number of channels in the Measurement Set.
 *
 * @details
 * Returns the number of channels in the Measurement Set.
 */
OSKAR_MS_EXPORT
unsigned int oskar_ms_num_channels(const oskar_MeasurementSet* p);

/**
 * @brief
 * Returns the number of polarisations in the Measurement Set.
 *
 * @details
 * Returns the number of polarisations in the Measurement Set.
 */
OSKAR_MS_EXPORT
unsigned int oskar_ms_num_pols(const oskar_MeasurementSet* p);

/**
 * @brief
 * Returns the number of rows in the main table.
 *
 * @details
 * Returns the number of rows in the main table.
 */
OSKAR_MS_EXPORT
unsigned int oskar_ms_num_rows(const oskar_MeasurementSet* p);

/**
 * @brief
 * Returns the number of stations in the Measurement Set.
 *
 * @details
 * Returns the number of stations in the Measurement Set.
 */
OSKAR_MS_EXPORT
unsigned int oskar_ms_num_stations(const oskar_MeasurementSet* p);

/**
 * @brief
 * Returns the phase centre type in the Measurement Set.
 *
 * @details
 * Returns the phase centre type in the Measurement Set.
 * 0 = Tracking (RA, Dec);
 * 1 = Drift scan (Azimuth, Elevation).
 */
OSKAR_MS_EXPORT
int oskar_ms_phase_centre_coord_type(const oskar_MeasurementSet* p);

/**
 * @brief
 * Returns the phase centre longitude in the Measurement Set.
 *
 * @details
 * Returns the phase centre longitude in the Measurement Set.
 */
OSKAR_MS_EXPORT
double oskar_ms_phase_centre_longitude_rad(const oskar_MeasurementSet* p);

/**
 * @brief
 * Returns the phase centre latitude in the Measurement Set.
 *
 * @details
 * Returns the phase centre latitude in the Measurement Set.
 */
OSKAR_MS_EXPORT
double oskar_ms_phase_centre_latitude_rad(const oskar_MeasurementSet* p);

/**
 * @brief
 * (DEPRECATED) Returns the phase centre RA in the Measurement Set.
 *
 * @details
 * (DEPRECATED) Returns the phase centre RA in the Measurement Set.
 */
OSKAR_MS_EXPORT
double oskar_ms_phase_centre_ra_rad(const oskar_MeasurementSet* p);

/**
 * @brief
 * (DEPRECATED) Returns the phase centre Dec in the Measurement Set.
 *
 * @details
 * (DEPRECATED) Returns the phase centre Dec in the Measurement Set.
 */
OSKAR_MS_EXPORT
double oskar_ms_phase_centre_dec_rad(const oskar_MeasurementSet* p);

/**
 * @brief
 * Sets the array centre.
 *
 * @details
 * Sets the ITRF coordinates of the array centre.
 *
 * @param[in] array_centre_itrf   The ITRF array centre coordinates, in metres.
 */
OSKAR_MS_EXPORT
void oskar_ms_set_array_centre(oskar_MeasurementSet* p,
        const double array_centre_itrf[3]);

/**
 * @brief
 * Sets the observation phase centre.
 *
 * @details
 * Sets the observation phase centre.
 *
 * The coordinate type can be either:
 * 0 = Tracking (RA, Dec);
 * 1 = Drift scan (Azimuth, Elevation).
 *
 * @param[in] coord_type     Coordinate type.
 * @param[in] longitude_rad  The longitude, in radians.
 * @param[in] latitude_rad   The latitude, in radians.
 */
OSKAR_MS_EXPORT
void oskar_ms_set_phase_centre(oskar_MeasurementSet* p, int coord_type,
        double longitude_rad, double latitude_rad);

/**
 * @brief Writes element positions to the PHASED_ARRAY table.
 *
 * @details
 * Adds the supplied list of element positions to the PHASED_ARRAY table.
 *
 * @param[in] station       The station index.
 * @param[in] num_elements  The number of element coordinates.
 * @param[in] x             The element x positions.
 * @param[in] y             The element y positions.
 * @param[in] z             The element z positions.
 * @param[in] matrix[9]     Local to ITRF transformation matrix.
 */
OSKAR_MS_EXPORT
void oskar_ms_set_element_coords(oskar_MeasurementSet* p, unsigned int station,
        unsigned int num_elements, const double* x, const double* y,
        const double* z, const double* matrix);

/**
 * @brief Writes station positions to the ANTENNA table.
 *
 * @details
 * Adds the supplied list of station positions to the ANTENNA table.
 *
 * @param[in] num_stations  The number of stations to add.
 * @param[in] x             The station x positions.
 * @param[in] y             The station y positions.
 * @param[in] z             The station z positions.
 */
OSKAR_MS_EXPORT
void oskar_ms_set_station_coords_d(oskar_MeasurementSet* p,
        unsigned int num_stations, const double* x, const double* y,
        const double* z);

OSKAR_MS_EXPORT
void oskar_ms_set_time_range(oskar_MeasurementSet* p);

/**
 * @brief
 * Returns the time increment in the Measurement Set.
 *
 * @details
 * Returns the time increment in the Measurement Set.
 */
OSKAR_MS_EXPORT
double oskar_ms_time_inc_sec(const oskar_MeasurementSet* p);

/**
 * @brief
 * Returns the start time in the Measurement Set.
 *
 * @details
 * Returns the start time in the Measurement Set.
 */
OSKAR_MS_EXPORT
double oskar_ms_time_start_mjd_utc(const oskar_MeasurementSet* p);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
