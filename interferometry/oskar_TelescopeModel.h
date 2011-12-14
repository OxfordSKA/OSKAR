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

#ifndef OSKAR_TELESCOPEMODEL_H_
#define OSKAR_TELESCOPEMODEL_H_

#include "oskar_global.h"
#include "station/oskar_StationModel.h"

/**
 * @struct oskar_TelescopeModel
 *
 * @brief
 *
 * @details
 */
struct oskar_TelescopeModel
{
    int num_stations;            /**< Number of stations in the model. */
    oskar_StationModel* station; /**< Array of station structures. */
    oskar_Mem station_x;         /**< Fixed x component of station coordinate. */
    oskar_Mem station_y;         /**< Fixed y component of station coordinate. */
    oskar_Mem station_z;         /**< Fixed z component of station coordinate. */
    int coord_units;             /**< Units of the x,y,z coordinates.*/
    int identical_stations;      /**< True if all stations are identical. */
    int disable_e_jones;         /**< If True, E-Jones is disabled. */
    int use_common_sky;          /**< True if all stations should use common source positions. */
    double ra0_rad;              /**< Right Ascension of phase centre, in radians. */
    double dec0_rad;             /**< Declination of phase centre, in radians. */
    double wavelength_metres;    /**< Current wavelength of observation, in metres. */
    double bandwidth_hz;         /**< Channel bandwidth, in Hz. */

#ifdef __cplusplus
    /* If C++, then provide constructors and methods. */
    /**
     * @brief Constructs a telescope model structure.
     *
     * @details
     * Constructs, initialises and allocates memory for a telescope model
     * data structure.
     *
     * @param[in] type         Array element type (OSKAR_SINGLE or OSKAR_DOUBLE).
     * @param[in] location     Memory location (OSKAR_LOCATION_CPU or OSKAR_LOCATION_GPU).
     * @param[in] num_stations Number of stations.
     */
    oskar_TelescopeModel(int type, int location, int num_stations = 0);

    /**
     * @brief Constructs a telescope model structure from an existing one.
     *
     * @details
     * Constructs a telescope model data structure by copying an existing one
     * to the specified location.
     *
     * @param[in] other Pointer to another structure to copy.
     * @param[in] location Memory location (OSKAR_LOCATION_CPU or OSKAR_LOCATION_GPU).
     */
    oskar_TelescopeModel(const oskar_TelescopeModel* other, int location);

    /**
     * @brief
     * Destroys the telescope structure, freeing any memory it occupies.
     *
     * @details
     * Destroys the telescope structure, freeing any memory it occupies.
     */
    ~oskar_TelescopeModel();

    /**
     * @brief
     * Loads a telescope coordinate file that specifies the station locations
     * with respect to the local tangent plane.
     *
     * @details
     * A telescope station coordinate file is an ASCII text file containing two
     * or three columns of comma- or space-separated values that represent the
     * station (x,y,z) coordinates in the local tangent plane. Each line
     * corresponds to the position of one station, and the z coordinate is
     * assumed to be zero if omitted.
     *
     * The coordinate system (ENU, or East-North-Up) is aligned so that the
     * x-axis points to the local geographic East, the y-axis to local
     * geographic North, and the z-axis to the local zenith. The origin is the
     * tangent point with the Earth's ellipsoid.
     *
     * The geodetic longitude and latitude of the origin must also be supplied.
     *
     * @param filename   File name path to a telescope coordinate file.
     * @param longitude  Telescope centre longitude, in radians.
     * @param latitude   Telescope centre latitude, in radians.
     * @param altitude   Telescope centre altitude, in metres.
     *
     * @return
     * This function returns a code to indicate if there were errors in execution:
     * - A return code of 0 indicates no error.
     * - A positive return code indicates a CUDA error.
     * - A negative return code indicates an OSKAR error.
     */
    int load_station_pos(const char* filename, double longitude,
            double latitude, double altitude);

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
     * @param[in] index    Index into station array to load.
     * @param[in] filename Name of the data file to load.
     *
     * @return
     * This function returns a code to indicate if there were errors in execution:
     * - A return code of 0 indicates no error.
     * - A positive return code indicates a CUDA error.
     * - A negative return code indicates an OSKAR error.
     */
    int load_station(int index, const char* filename);

    /**
     * @brief Returns the location of all arrays in the structure, or an error
     * code if the locations are inconsistent.
     */
    int location() const;

    /**
     * @brief Multiplies all coordinates by the wavenumber.
     *
     * @param[in] frequency_hz Frequency, in Hz.
     */
    int multiply_by_wavenumber(double frequency_hz);

    /**
     * @brief
     * Resizes the telescope model structure.
     *
     * @details
     * Resizes the telescope model structure.
     *
     * @param n_stations New number of stations.
     */
    int resize(int n_stations);

    /**
     * @brief Returns the base type of all arrays in the structure, or an error
     * code if the types are inconsistent.
     */
    int type() const;

    /**
     * @brief Returns the number of baselines.
     */
    int num_baselines() const
    { return ((num_stations * (num_stations - 1)) / 2); }

#endif
};

typedef struct oskar_TelescopeModel oskar_TelescopeModel;

#endif /* OSKAR_TELESCOPEMODEL_H_ */
