/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#ifndef OSKAR_PRIVATE_TELESCOPE_H_
#define OSKAR_PRIVATE_TELESCOPE_H_

#include <oskar_mem.h>
#include <oskar_station.h>

/**
 * @struct oskar_Telescope
 *
 * @brief
 *
 * @details
 */
struct oskar_Telescope
{
    /* Private structure meta-data. */
    int precision;
    int location;

    /* Fundamental properties that don't depend on stations. */
    double longitude_rad;        /**< Geodetic longitude of telescope, in radians. */
    double latitude_rad;         /**< Geodetic latitude of telescope, in radians. */
    double altitude_m;           /**< Altitude of telescope above ellipsoid, in metres. */
    double ra0_rad;              /**< Right Ascension of phase centre, in radians. */
    double dec0_rad;             /**< Declination of phase centre, in radians. */
    double bandwidth_hz;         /**< Channel bandwidth, in Hz. */
    double time_average_sec;     /**< Time average smearing interval, in sec. */
    double wavenumber;           /**< Current wavenumber, if coordinates are in radians (private). */

    /* Station data. */
    int num_stations;            /**< Number of stations in the model. */
    oskar_Station** station;     /**< Array of station structure handles. */
    oskar_Mem station_x;         /**< Fixed x component of station coordinate. */
    oskar_Mem station_y;         /**< Fixed y component of station coordinate. */
    oskar_Mem station_z;         /**< Fixed z component of station coordinate. */
    oskar_Mem station_x_hor;     /**< Fixed x component of station coordinate (horizon plane). */
    oskar_Mem station_y_hor;     /**< Fixed y component of station coordinate (horizon plane). */
    oskar_Mem station_z_hor;     /**< Fixed z component of station coordinate (horizon plane). */
    int max_station_size;        /**< Maximum station size (number of elements) */
    int max_station_depth;       /**< Maximum station depth. */
    int coord_units;             /**< Units of the x,y,z coordinates .*/
    int identical_stations;      /**< True if all stations are identical. */
    int use_common_sky;          /**< True if all stations should use common source positions. */

    /** Random seed for time-variable station element errors (amplitude and phase). */
    int seed_time_variable_station_element_errors;
};

#ifndef OSKAR_TELESCOPE_TYPEDEF_
#define OSKAR_TELESCOPE_TYPEDEF_
typedef struct oskar_Telescope oskar_Telescope;
#endif /* OSKAR_TELESCOPE_TYPEDEF_ */

#endif /* OSKAR_PRIVATE_TELESCOPE_H_ */
