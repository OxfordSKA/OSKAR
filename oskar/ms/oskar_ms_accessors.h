/*
 * Copyright (c) 2011-2017, The University of Oxford
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

#ifndef OSKAR_MS_ACCESSORS_H_
#define OSKAR_MS_ACCESSORS_H_

/**
 * @file oskar_ms_accessors.h
 */

#include <oskar_global.h>
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
 * Returns the phase centre RA in the Measurement Set.
 *
 * @details
 * Returns the phase centre RA in the Measurement Set.
 */
OSKAR_MS_EXPORT
double oskar_ms_phase_centre_ra_rad(const oskar_MeasurementSet* p);

/**
 * @brief
 * Returns the phase centre Dec in the Measurement Set.
 *
 * @details
 * Returns the phase centre Dec in the Measurement Set.
 */
OSKAR_MS_EXPORT
double oskar_ms_phase_centre_dec_rad(const oskar_MeasurementSet* p);

/**
 * @brief
 * Sets the observation phase centre.
 *
 * @details
 * Sets the observation phase centre.
 *
 * @param[in] coord_type     Coordinate type (reserved; currently ignored).
 * @param[in] longitude_rad  The longitude (Right Ascension), in radians.
 * @param[in] latitude_rad   The latitude (Declination), in radians.
 */
OSKAR_MS_EXPORT
void oskar_ms_set_phase_centre(oskar_MeasurementSet* p, int coord_type,
        double longitude_rad, double latitude_rad);

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
void oskar_ms_set_station_coords_f(oskar_MeasurementSet* p,
        unsigned int num_stations, const float* x, const float* y,
        const float* z);

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

#endif /* OSKAR_MS_ACCESSORS_H_ */
