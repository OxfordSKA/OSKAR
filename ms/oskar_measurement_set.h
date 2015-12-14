/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#ifndef OSKAR_MEASUREMENT_SET_H_
#define OSKAR_MEASUREMENT_SET_H_

/**
 * @file oskar_measurement_set.h
 */

#include <oskar_global.h>
#include <stddef.h>

/* Public interface. */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_MeasurementSet;
#ifndef OSKAR_MEASUREMENT_SET_TYPEDEF_
#define OSKAR_MEASUREMENT_SET_TYPEDEF_
typedef struct oskar_MeasurementSet oskar_MeasurementSet;
#endif /* OSKAR_MEASUREMENT_SET_TYPEDEF_ */

/**
 * @brief Adds the run log to the HISTORY table.
 *
 * @details
 * Adds the supplied run log string to the HISTORY table.
 * The log string is split into lines, and each is added as its own
 * HISTORY entry. The origin of each will be "LOG".
 *
 * @param[in] str   The log file, as a single string.
 * @param[in] size  The length of the string.
 */
OSKAR_MS_EXPORT
void oskar_ms_add_log(oskar_MeasurementSet* p, const char* str, size_t size);

/**
 * @brief Adds the contents of the settings file to the HISTORY table.
 *
 * @details
 * Adds the supplied settings file string to the HISTORY table.
 * The settings string is split into lines, and these are entered to
 * the APP_PARAMS column as a single HISTORY entry.
 * The history message will read "OSKAR settings file", and the origin
 * will be "SETTINGS".
 *
 * @param[in] str   The settings file, as a single string.
 * @param[in] size  The length of the string.
 */
OSKAR_MS_EXPORT
void oskar_ms_add_settings(oskar_MeasurementSet* p,
        const char* str, size_t size);

/**
 * @brief Adds scratch columns CORRECTED_DATA and MODEL_DATA.
 *
 * @details
 * Add the scratch columns CORRECTED_DATA and MODEL_DATA to the Measurement Set
 * main table.
 *
 * @param[in] add_model     If true, add the MODEL_DATA column.
 * @param[in] add_corrected If true, add the CORRECTED_DATA column.
 */
OSKAR_MS_EXPORT
void oskar_ms_add_scratch_columns(oskar_MeasurementSet* p,
        int add_model, int add_corrected);

/**
 * @brief Copies data from one column to another.
 *
 * @details
 * Copies data from one column to another.
 * Both columns must contain arrays of the same shape and data type.
 *
 * @param[in] source     Name of source column to copy.
 * @param[in] dest       Name of destination column to copy into.
 */
OSKAR_MS_EXPORT
void oskar_ms_copy_column(oskar_MeasurementSet* p,
        const char* source, const char* dest);

/**
 * @brief Adds station positions to the ANTENNA table.
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
 * @brief Adds station positions to the ANTENNA table.
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

/**
 * @brief Closes the Measurement Set.
 *
 * @details
 * Closes the Measurement Set and flushes any pending write operations to disk.
 */
OSKAR_MS_EXPORT
void oskar_ms_close(oskar_MeasurementSet* p);

/**
 * @brief Creates a new Measurement Set.
 *
 * @details
 * Creates a new, empty Measurement Set with the given name.
 *
 * @param[in] filename        The filename to use.
 * @param[in] ra_rad          The right ascension of the phase centre, in rad.
 * @param[in] dec_rad         The declination of the phase centre, in rad.
 * @param[in] num_pols        The number of polarisations (1, 2 or 4).
 * @param[in] num_channels    The number of channels in the band.
 * @param[in] ref_freq        The frequency at the centre of channel 0, in Hz.
 * @param[in] chan_width      The width of each channel in Hz.
 * @param[in] num_stations    The number of antennas/stations.
 * @param[in] write_autocorr  If set, write auto-correlation data.
 * @param[in] write_crosscorr If set, write cross-correlation data.
 */
OSKAR_MS_EXPORT
oskar_MeasurementSet* oskar_ms_create(const char* filename, double ra_rad,
        double dec_rad, unsigned int num_pols, unsigned int num_channels,
        double ref_freq, double chan_width, unsigned int num_stations,
        int write_autocorr, int write_crosscorr);

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
 * Returns the number of channels in the Measurement Set.
 *
 * @details
 * Returns the number of channels in the Measurement Set.
 */
OSKAR_MS_EXPORT
unsigned int oskar_ms_num_channels(const oskar_MeasurementSet* p);

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
 * @brief Opens an existing Measurement Set.
 *
 * @details
 * Opens an existing Measurement Set.
 */
OSKAR_MS_EXPORT
oskar_MeasurementSet* oskar_ms_open(const char* filename);

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
 * Returns the reference frequency in the Measurement Set.
 *
 * @details
 * Returns the reference frequency in the Measurement Set.
 */
OSKAR_MS_EXPORT
double oskar_ms_ref_freq_hz(const oskar_MeasurementSet* p);

/**
 * @details
 * Writes visibility data to the main table.
 *
 * @details
 * This function writes the given block of visibility data to the main table of
 * the Measurement Set. The dimensionality of the complex \p vis data block
 * is num_pols * num_channels * num_baselines, with num_pols the
 * fastest varying dimension, then num_channels, and finally num_baselines.
 *
 * Each row of the main table holds data from a single baseline for a
 * single time stamp. The data block is repeated as many times as necessary.
 *
 * The time is given in units of (MJD) * 86400, i.e. seconds since
 * Julian date 2400000.5.
 *
 * The layout of \p vis corresponding to a three-element interferometer with
 * four polarisations and two channels would be (for C-ordered memory):
 *
 * ant0-1
 *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
 *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
 * ant0-2
 *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
 *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
 * ant1-2
 *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
 *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
 *
 * @param[in] start_row     The start row index of the Measurement Set.
 * @param[in] num_baselines Number of rows to add to the main table (see note).
 * @param[in] u             Baseline u-coordinates, in metres.
 * @param[in] v             Baseline v-coordinate, in metres.
 * @param[in] w             Baseline w-coordinate, in metres.
 * @param[in] vis           Matrix of complex visibilities per row (see note).
 * @param[in] ant1          Indices of antenna 1 for each baseline.
 * @param[in] ant2          Indices of antenna 2 for each baseline.
 * @param[in] exposure      The exposure length per visibility, in seconds.
 * @param[in] interval      The interval length per visibility, in seconds.
 * @param[in] time          Timestamp of visibility block.
 */
OSKAR_MS_EXPORT
void oskar_ms_write_all_for_time_d(oskar_MeasurementSet* p,
        unsigned int start_row, unsigned int num_baselines,
        const double* u, const double* v, const double* w, const double* vis,
        const int* ant1, const int* ant2, double exposure, double interval,
        double time);

/**
 * @details
 * Writes visibility data to the main table.
 *
 * @details
 * This function writes the given block of visibility data to the main table of
 * the Measurement Set. The dimensionality of the complex \p vis data block
 * is num_pols * num_channels * num_baselines, with num_pols the
 * fastest varying dimension, then num_channels, and finally num_baselines.
 *
 * Each row of the main table holds data from a single baseline for a
 * single time stamp. The data block is repeated as many times as necessary.
 *
 * The time is given in units of (MJD) * 86400, i.e. seconds since
 * Julian date 2400000.5.
 *
 * The layout of \p vis corresponding to a three-element interferometer with
 * four polarisations and two channels would be (for C-ordered memory):
 *
 * ant0-1
 *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
 *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
 * ant0-2
 *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
 *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
 * ant1-2
 *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
 *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
 *
 * @param[in] start_row     The start row index of the Measurement Set.
 * @param[in] num_baselines Number of rows to add to the main table (see note).
 * @param[in] u             Baseline u-coordinates, in metres.
 * @param[in] v             Baseline v-coordinate, in metres.
 * @param[in] w             Baseline w-coordinate, in metres.
 * @param[in] vis           Matrix of complex visibilities per row (see note).
 * @param[in] ant1          Indices of antenna 1 for each baseline.
 * @param[in] ant2          Indices of antenna 2 for each baseline.
 * @param[in] exposure      The exposure length per visibility, in seconds.
 * @param[in] interval      The interval length per visibility, in seconds.
 * @param[in] time          Timestamp of visibility block.
 */
OSKAR_MS_EXPORT
void oskar_ms_write_all_for_time_f(oskar_MeasurementSet* p,
        unsigned int start_row, unsigned int num_baselines,
        const float* u, const float* v, const float* w, const float* vis,
        const int* ant1, const int* ant2, double exposure, double interval,
        double time);

/**
 * @brief Sets the number of rows in the main table.
 *
 * @details
 * Sets the number of rows in the main table to be at least as large
 * as the value given.
 */
OSKAR_MS_EXPORT
void oskar_ms_set_num_rows(oskar_MeasurementSet* p, unsigned int num);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEASUREMENT_SET_H_ */
