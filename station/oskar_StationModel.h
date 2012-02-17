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

#ifndef OSKAR_STATION_MODEL_H_
#define OSKAR_STATION_MODEL_H_

/**
 * @file oskar_StationModel.h
 */

#include "oskar_global.h"
#include "utility/oskar_Mem.h"
#include "station/oskar_ElementModel.h"

/* Forward declaration. */
struct oskar_StationModel;
typedef struct oskar_StationModel oskar_StationModel;

struct oskar_StationModel
{
    int num_elements;
    oskar_StationModel* child;  /**< NULL when there are no child stations. */
    oskar_StationModel* parent; /**< Pointer to station's parent (NULL if none). */
    oskar_ElementModel* element_pattern; /**< NULL if there are child stations. */

    /* Station element data. */
    oskar_Mem x;      /**< x-position wrt local horizon, toward the East. */
    oskar_Mem y;      /**< y-position wrt local horizon, toward the North. */
    oskar_Mem z;      /**< z-position wrt local horizon, toward the zenith. */
    int coord_units;  /**< Units of the x,y,z coordinates.*/

    int apply_weight; /**< Bool switch to toggle complex element weight vector (default false) */
    oskar_Mem weight; /**< Element complex weight (set to 1 unless apodisation). */

    int apply_element_errors; /**< Bool switch to toggle element gain and phase errors (default false) */
    oskar_Mem amp_gain;       /**< Per element amplitude gain factor (default 1.0) */
    oskar_Mem amp_gain_error; /**< Standard deviation of per element random amplitude gain factor (default 0.0) */
    oskar_Mem phase_offset;   /**< Per element systematic phase offset, in radians (default 0.0) */
    oskar_Mem phase_error;    /**< Standard deviation of per element random phase offset, in radians (default 0.0) */

    oskar_Mem total_receiver_noise; /**< Total receiver noise stddev as a
                                         function of frequency, in Jy */

    /* Other station data. */
    double longitude_rad;   /**< Geodetic longitude of station, in radians. */
    double latitude_rad;    /**< Geodetic latitude of station, in radians. */
    double altitude_metres; /**< Altitude of station above ellipsoid, in metres. */
    double ra0_rad;         /**< Right ascension of beam phase centre, in radians. */
    double dec0_rad;        /**< Declination of beam phase centre, in radians. */
    int single_element_model; /**< True if using a single common element pattern. */
    int normalise_beam; /**< True if the station beam should be normalised by the number of antennas. */
    int bit_depth;    /**< Not implemented! */

#ifdef __cplusplus
    /* If C++, provide constructors and methods. */
    /**
     * @brief Constructs an empty station model structure.
     *
     * @details
     * Constructs an empty station model structure.
     */
    oskar_StationModel(int type, int location, int n_elements = 0);

    /**
     * @brief Constructs a copy of another station model structure.
     *
     * @details
     * Copies an existing station model structure to the specified location.
     */
    oskar_StationModel(const oskar_StationModel* other, int location);

    /**
     * @brief Destroys the station model structure.
     *
     * @details
     * Destroys the station model structure, freeing any memory it uses.
     */
    ~oskar_StationModel();

    /**
     * @brief Copies this station model structure to another.
     *
     * @details
     * Copies the memory in this station model to that in another.
     */
    int copy_to(oskar_StationModel* other);

    /**
     * @brief
     * Loads the station element data from a text file.
     *
     * @details
     * This function loads station element (antenna) data from a comma- or
     * space-separated text file. Each line contains data for one element of the
     * station.
     *
     * The file may have the following columns, in the following order:
     * - Element x-position, in metres.
     * - Element y-position, in metres.
     * - Element z-position, in metres (default 0).
     * - Element multiplicative weight (real part, default 1).
     * - Element multiplicative weight (imaginary part, default 0).
     * - Element amplitude gain factor (default 1).
     * - Element amplitude gain error (default 0).
     * - Element phase offset in degrees (default 0).
     * - Element phase error in degrees (default 0).
     *
     * Only the first two columns are required to be present.
     *
     * The coordinate system (ENU, or East-North-Up) is aligned so that the
     * x-axis points to the local geographic East, the y-axis to local
     * geographic North, and the z-axis to the local zenith.
     *
     * @param[out] station Pointer to destination data structure to fill.
     * @param[in] filename Name of the data file to load.
     *
     * @return
     * This function returns a code to indicate if there were errors in execution:
     * - A return code of 0 indicates no error.
     * - A positive return code indicates a CUDA error.
     * - A negative return code indicates an OSKAR error.
     */
    int load(const char* filename);

    /**
     * @brief Returns the location of all arrays in the structure, or an error
     * code if the locations are inconsistent.
     */
    int location() const;

    /**
     * @brief Resizes the station structure.
     *
     * @details
     * Resizes the memory arrays held by the station structure.
     *
     * @param[in] n_elements The new number of elements the arrays can hold.
     */
    int resize(int n_elements);

    /**
     * @brief Scales station coordinates from metres to wavenumber units.
     *
     * @param[in] frequency_hz Frequency, in Hz.
     */
    int multiply_by_wavenumber(double frequency_hz);

    /**
     * @brief Returns the base type of all arrays in the structure, or an error
     * code if the types are inconsistent.
     */
    int type() const;

    /**
     * @brief Returns the oskar_Mem type ID of station coordinates or
     * an error code if the coordinate type is invalid.
     */
    int coord_type() const;

    /**
     * @brief Returns the oskar_Mem location ID of the station coordinates or
     * an error code if the coordinate type is invalid.
     */
    int coord_location() const;
#endif
};

#endif /* OSKAR_STATION_MODEL_H_ */
