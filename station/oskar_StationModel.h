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

#ifdef __cplusplus
extern "C"
#endif
struct oskar_StationModel
{
    int n_elements;
    oskar_StationModel* child;  /**< NULL when there are no child stations. */
    oskar_StationModel* parent; /**< Pointer to station's parent (NULL if none). */
    oskar_ElementModel* element_pattern; /**< NULL if there are child stations. */

    /* Station element data. */
    oskar_Mem x;      /**< x-position wrt local horizon, toward the East. */
    oskar_Mem y;      /**< y-position wrt local horizon, toward the North. */
    oskar_Mem z;      /**< z-position wrt local horizon, toward the zenith. */
    int coord_units;  /**< Units of the x,y,z coordinates.*/
    oskar_Mem weight; /**< Element complex weight (set to 1 unless apodisation). */
    oskar_Mem amp_gain;
    oskar_Mem amp_error;
    oskar_Mem phase_offset;
    oskar_Mem phase_error;

    /* Other station data. */
    double longitude; /**< Geodetic longitude of station, in radians. */
    double latitude;  /**< Geodetic latitude of station, in radians. */
    double altitude;  /**< Altitude of station above ellipsoid, in metres. */
    double ra0;       /**< Right ascension of beam phase centre, in radians. */
    double dec0;      /**< Declination of beam phase centre, in radians. */
    int single_element_model; /**< True if using a single common element pattern. */
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
    int scale_coords_to_wavenumbers(const double frequency_hz);

    /**
     * @brief Returns the oskar_Mem type ID of station coordinates or
     * an error code if the coordinate type is invalid.
     */
    int coord_type() const;

    /**
     * @brief Returns the oskar_Mem locaiton ID of the station coordinates or
     * an error code if the coordinate type is invalid.
     */
    int coord_location() const;
#endif
};


/* DEPRECATED */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_StationModel_d
{
    unsigned num_antennas;
    double*  antenna_x;
    double*  antenna_y;
};
typedef struct oskar_StationModel_d oskar_StationModel_d;

struct oskar_StationModel_f
{
    unsigned num_antennas;
    float*   antenna_x;
    float*   antenna_y;
};
typedef struct oskar_StationModel_f oskar_StationModel_f;


OSKAR_EXPORT
void oskar_station_model_copy_to_device_d(const oskar_StationModel_d* h_stations,
        const unsigned num_stations, oskar_StationModel_d* hd_stations);

OSKAR_EXPORT
void oskar_station_model_copy_to_device_f(const oskar_StationModel_f* h_stations,
        const unsigned num_stations, oskar_StationModel_f* hd_stations);

OSKAR_EXPORT
void oskar_station_model_scale_coords_d(const unsigned num_stations,
        oskar_StationModel_d* hd_stations, const double value);

OSKAR_EXPORT
void oskar_station_model_scale_coords_f(const unsigned num_stations,
        oskar_StationModel_f* hd_stations, const float value);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_STATION_MODEL_H_ */
