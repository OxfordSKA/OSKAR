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

#ifndef OSKAR_PRIVATE_STATION_H_
#define OSKAR_PRIVATE_STATION_H_

/**
 * @file private_station.h
 */

#include <oskar_mem.h>
#include <oskar_ElementModel.h>
#include <oskar_SystemNoiseModel.h>

/* Forward declaration. */
struct oskar_Station;

#ifndef OSKAR_STATION_TYPEDEF_
#define OSKAR_STATION_TYPEDEF_
typedef struct oskar_Station oskar_Station;
#endif /* OSKAR_STATION_TYPEDEF_ */

struct oskar_Station
{
    /* Private structure meta-data. */
    int precision;
    int location;

    /* Data common to all station types -------------------------------------*/
    int station_type;            /**< Type of the station (enumerator). */
    double longitude_rad;        /**< Geodetic longitude of station, in radians. */
    double latitude_rad;         /**< Geodetic latitude of station, in radians. */
    double altitude_m;           /**< Altitude of station above ellipsoid, in metres. */
    double beam_longitude_rad;   /**< Longitude of beam phase centre, in radians. */
    double beam_latitude_rad;    /**< Latitude of beam phase centre, in radians. */
    int beam_coord_type;         /**< Enumerator describing beam spherical coordinate type (from oskar_global.h). */
    oskar_SystemNoiseModel noise;

    /* Data used only for Gaussian beam stations  ---------------------------*/
    double gaussian_beam_fwhm_rad; /**< FWHM of Gaussian station beam, in degrees. */

    /* Data used only for aperture array stations ---------------------------*/
    int num_elements;            /**< Number of antenna elements in the station (auto determined). */
    int num_element_types;       /**< Number of element types (this is the size of element_pattern array). */
    int use_polarised_elements;  /**< True if station elements are polarised. */
    int normalise_beam;          /**< True if the station beam should be normalised by the number of antennas. */
    int enable_array_pattern;    /**< True if the array factor should be evaluated. */
    int single_element_model;    /**< True if using a single common element pattern. */
    int array_is_3d;             /**< True if array is 3-dimensional (auto determined; default false). */
    int apply_element_errors;    /**< True if element gain and phase errors should be applied (auto determined; default false). */
    int apply_element_weight;    /**< True if weights should be modified by user-supplied complex beamforming weights (auto determined; default false). */
    int coord_units;             /**< Units of the x,y,z coordinates (auto determined). */
    double wavenumber;           /**< Current wavenumber, if coordinates are in radians (private). */
    double orientation_x;        /**< Orientation azimuth of nominal x dipole axis, in radians. */
    double orientation_y;        /**< Orientation azimuth of nominal y dipole axis, in radians. */
    oskar_Mem x_signal;          /**< Tangent-plane x-position, toward the East. */
    oskar_Mem y_signal;          /**< Tangent-plane y-position, toward the North. */
    oskar_Mem z_signal;          /**< Tangent-plane z-position, toward the zenith. */
    oskar_Mem x_weights;         /**< Tangent-plane x-position used for weights computation, toward the East. */
    oskar_Mem y_weights;         /**< Tangent-plane y-position used for weights computation, toward the North. */
    oskar_Mem z_weights;         /**< Tangent-plane z-position used for weights computation, toward the zenith. */
    oskar_Mem gain;              /**< Per-element gain factor (default 1.0) */
    oskar_Mem gain_error;        /**< Standard deviation of per-element time-variable gain factor (default 0.0) */
    oskar_Mem phase_offset;      /**< Per-element systematic phase offset, in radians (default 0.0) */
    oskar_Mem phase_error;       /**< Standard deviation of per-element time-variable phase offset, in radians (default 0.0) */
    oskar_Mem weight;            /**< Element complex weight (set to 1.0, 0.0 unless using apodisation). */
    oskar_Mem cos_orientation_x; /**< Cosine azimuth of x dipole axis (default 0.0). */
    oskar_Mem sin_orientation_x; /**< Sine azimuth of x dipole axis (default 1.0) */
    oskar_Mem cos_orientation_y; /**< Cosine azimuth of y dipole axis (default 1.0) */
    oskar_Mem sin_orientation_y; /**< Sine azimuth of y dipole axis (default 0.0) */
    oskar_Mem element_type;      /**< Integer array of element types (default 0). */
    oskar_Station** child;       /**< Array of child station handles (pointer is NULL if there are none). */
    oskar_ElementModel* element_pattern; /**< Array of element models per element type (pointer is NULL if there are child stations). */
};

#endif /* OSKAR_PRIVATE_STATION_H_ */
