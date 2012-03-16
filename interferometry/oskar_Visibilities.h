/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_VISIBILITIES_H_
#define OSKAR_VISIBILITIES_H_

/**
 * @file oskar_Visibilities.h
 */


#include "oskar_global.h"
#include "utility/oskar_Mem.h"
#include "interferometry/oskar_TelescopeModel.h"
#include <stdlib.h>

#define OSKAR_VIS_FILE_ID 0x1F4F

/**
 * @brief Structure to hold visibility data.
 *
 * @details
 * Visibility data consists of a number of samples each of which consist of
 * a set of baselines coordinates (u,v,w) in wave-numbers and one or more complex
 * visibility amplitudes.
 *
 * Baseline coordinates are stored as three separate arrays of length corresponding
 * to the number of visibility samples in the data set. Coordinates are real
 * valued and can be either double or single precision. Coordinates are ordered
 * by channel ID, time and baseline where channel is the slowest varying dimension
 * and baseline is the fastest varying. This is coordinate ordering is chosen
 * to for optimal memory write performance in OSKAR where channels are simulated
 * on the outer processing loop.
 *
 * Axis order: channel(slowest) -> time -> baseline (fastest)
 *
 * Visibility amplitudes are sorted as an array of complex scalar or 2-by-2 matrix
 * elements which can be either double or single precision according to the
 * amplitude type ID. The dimension ordering is the same as that used for
 * the baseline coordinate arrays. The use of scalar or matrix elements in the
 * array determines the number of polarisations stored in the data. Scalar elements
 * represent a single polarisation axis whereas matrix elements store 4 polarisations
 * per data sample.
 *
 * NOTE Consider meta-data fields for frequency and time under review w.r.t.
 * names and interface for setting.
 * NOTE Consider storing a coord_type_id for the baseline coordinates similar
 * to the station model.
 */
struct oskar_Visibilities
{
    oskar_Mem settings_path;     /**< Path to settings file. */
    int num_channels;            /**< Number of frequency channels. */
    int num_times;               /**< Number of time samples. */
    int num_baselines;           /**< Number of baselines. */
    double freq_start_hz;        /**< Start Frequency, in Hz. */
    double freq_inc_hz;          /**< Frequency increment, in Hz. */
    double time_start_mjd_utc;   /**< Start time in MJD, UTC */
    double time_inc_seconds;     /**< Time increment, in seconds. */
    double channel_bandwidth_hz; /**< Frequency channel bandwidth, in Hz */
    double phase_centre_ra_deg;  /**< Pointing phase centre RA, in degrees*/
    double phase_centre_dec_deg; /**< Pointing phase centre Dec, in degrees */

    oskar_Mem sky_noise_stddev;  /**< Standard deviation corresponding to all
                                      sky noise, per channel. */

    oskar_Mem uu_metres;         /**< Baseline coordinates, in metres. */
    oskar_Mem vv_metres;         /**< Baseline coordinates, in metres. */
    oskar_Mem ww_metres;         /**< Baseline coordinates, in metres. */
    oskar_Mem amplitude;         /**< Complex visibility amplitude. */

#ifdef __cplusplus
    /**
     * @brief Constructs a visibility structure according to the specified
     * dimensions, type and location.
     *
     * @details
     * Allowed values of the \p amp_type parameter are
     * - OSKAR_SINGLE_COMPLEX
     * - OSKAR_DOUBLE_COMPLEX
     * - OSKAR_SINGLE_COMPLEX_MATRIX
     * - OSKAR_DOUBLE_COMPLEX_MATRIX
     *
     * Allowed values of the \p location parameter are
     * - OSKAR_LOCATION_CPU
     * - OSKAR_LOCATION_GPU
     *
     * The number of polarisations is determined by the choice of matrix or
     * scalar amplitude types. Matrix amplitude types represent 4 polarisation
     * dimensions whereas scalar types represent a single polarisation.
     *
     * Note
     *  - Channels is the slowest varying dimension.
     *  - Baselines is the fastest varying dimension.
     *
     * @param amp_type       OSKAR type ID for visibility amplitudes.
     * @param location       OSKAR memory location ID for data in the structure.
     * @param num_channels   Number of frequency channels.
     * @param num_times      Number of visibility time snapshots.
     * @param num_baselines  Number of baselines.
     */
    oskar_Visibilities(int amp_type = OSKAR_SINGLE_COMPLEX_MATRIX,
            int location = OSKAR_LOCATION_CPU, int num_channels = 0,
            int num_times = 0, int num_baselines = 0);

    /**
     * @brief Constructs a visibility structure at the specified location
     * by copying it from the specified visibility structure.
     *
     * @details
     * This is effectively a copy constructor however the location of the
     * copy created by this call can be specified. This can be used for
     * example to create a copy of an visibility structure in a different
     * memory location.
     *
     * @param other     Visibility structure to copy.
     * @param location  Location to construct the new visibility structure.
     */
    oskar_Visibilities(const oskar_Visibilities* other, int location);

    /**
     * @brief Destructor.
     */
    ~oskar_Visibilities();

    /**
     * @brief Clears contents of the visibility structure.
     *
     * @details
     * Clears the memory contents of the visibility data.
     *
     * @return An error code.
     */
    int clear_contents();

    /**
     * @brief Writes the contents of visibility structure to an OSKAR
     * visibility dump file of the specified file name. This is a simple
     * custom binary format.
     *
     * @details
     * Note: This function currently requires the visibility structure memory
     * to reside on the CPU.
     *
     * @param filename The filename to write to.
     *
     * @return An error code.
     */
    int write(const char* filename);

    /**
     * @brief Returns a new visibility structure by reading the specified file.
     *
     * @details
     * Note: The loaded visibility structure will reside on the CPU.
     *
     * @param[in]  filename The filename to read from.
     * @param[out] status   Error code.
     *
     * @return A pointer to the new visibility structure.
     */
    static oskar_Visibilities* read(const char* filename, int* status = NULL);

    /**
     * @brief Resize the memory in the visibility structure to the specified
     * dimensions.
     *
     * @param num_channels   Number of frequency channels.
     * @param num_times      Number of visibility time snapshots.
     * @param num_baselines  Number of baselines.
     *
     * @return An error code.
     */
    int resize(int num_channels, int num_times, int num_baselines);

    /**
     * @brief Initialises the memory in the visibility structure
     * to the specified dimensions, type and location.
     *
     * @details
     * Warning: This function will erase any existing values in the visibility
     * structure.
     *
     * @param amp_type       OSKAR type ID for visibility amplitudes.
     * @param location       OSKAR memory location ID for data in the structure.
     * @param num_channels   Number of frequency channels.
     * @param num_times      Number of visibility time snapshots.
     * @param num_baselines  Number of baselines.
     *
     * @return An error code.
     */
    int init(int amp_type, int location, int num_channels, int num_times,
            int num_baselines);

    /**
     * @brief Returns an oskar_Mem pointer (non ownership) for the channel
     * amplitudes of the specified channel.
     *
     * @param vis_amps oskar_Mem pointer to the amplitudes for the specified
     *                 channel.
     * @param channel  Channel index for requested amplitudes
     *
     * @return An error code.
     */
    int get_channel_amps(oskar_Mem* vis_amps, int channel);

    /**
     * @brief Evaluates the sky noise standard deviation for the addition of
     * a gaussian sky noise component to the visibility amplitudes.
     *
     * @details
     * NOTE: the interface to this may change!
     *
     * @param telescope         OSKAR telescope model structure.
     * @param spectral_index    Spectral index (used to evaluate the sky noise
     *                          frequency response, usually +0.75)
     *
     * @return An error code.
     */
    int evaluate_sky_noise_stddev(const oskar_TelescopeModel* telescope,
            double spectral_index);

    /**
     * @brief Adds a frequency dependent gaussian noise component to the
     * visibilities, typically corresponding to sky noise.
     *
     * @details
     * NOTE: the interface to this may change!
     *
     * @param[in] stddev  Array (length num_channels) of the standard deviation
     *                    of the noise to add, in Janskys.
     *
     * @return An error code.
     */
    int add_sky_noise(const double* stddev, unsigned seed = 666);

    /**
     * @brief Returns the number of baseline u,v,w coordinates.
     */
    int num_coords() const
    { return num_times * num_baselines; }

    /**
     * @brief Returns the number of visibility amplitudes.
     */
    int num_amps() const
    { return num_channels * num_times * num_baselines; }

    /**
     * @brief Returns the OSKAR location ID of the visibilities structure.
     */
    int location() const;

    /**
     * @brief Returns the number of polarisations for each visibility sample.
     */
    int num_polarisations() const
    { return amplitude.is_scalar() ? 1 : 4; }
#endif
};

typedef struct oskar_Visibilities oskar_Visibilities;

/* To maintain binary compatibility, do not change the values
 * in the lists below. */
enum {
    OSKAR_VIS_TAG_NUM_CHANNELS = 1,
    OSKAR_VIS_TAG_NUM_TIMES = 2,
    OSKAR_VIS_TAG_NUM_BASELINES = 3,
    OSKAR_VIS_TAG_DIMENSION_ORDER = 4,
    OSKAR_VIS_TAG_COORD_TYPE = 5,
    OSKAR_VIS_TAG_AMP_TYPE = 6,
    OSKAR_VIS_TAG_FREQ_START_HZ = 7,
    OSKAR_VIS_TAG_FREQ_INC_HZ = 8,
    OSKAR_VIS_TAG_TIME_START_MJD_UTC = 9,
    OSKAR_VIS_TAG_TIME_INC_SEC = 10,
    OSKAR_VIS_TAG_POL_TYPE = 11,
    OSKAR_VIS_TAG_BASELINE_COORD_UNIT = 12,
    OSKAR_VIS_TAG_BASELINE_UU = 13,
    OSKAR_VIS_TAG_BASELINE_VV = 14,
    OSKAR_VIS_TAG_BASELINE_WW = 15,
    OSKAR_VIS_TAG_AMPLITUDE = 16
};

/* Do not change the values below - these are merely dimension labels, not the
 * actual dimension order. */
enum {
    OSKAR_VIS_DIM_CHANNEL = 0,
    OSKAR_VIS_DIM_TIME = 1,
    OSKAR_VIS_DIM_BASELINE = 2,
    OSKAR_VIS_DIM_POLARISATION = 3
};

enum {
    OSKAR_VIS_POL_TYPE_NONE = 0,
    OSKAR_VIS_POL_TYPE_LINEAR = 1
};

enum {
    OSKAR_VIS_BASELINE_COORD_UNIT_METRES = 1
};

#endif /* OSKAR_VISIBILITIES_H_ */
