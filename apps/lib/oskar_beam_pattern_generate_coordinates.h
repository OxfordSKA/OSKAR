/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#ifndef OSKAR_BEAM_PATTERN_GENERATE_COORDINATES_H_
#define OSKAR_BEAM_PATTERN_GENERATE_COORDINATES_H_

/**
 * @file oskar_beam_pattern_generate_coordinates.h
 */

#include <oskar_global.h>
#include <oskar_mem.h>
#include <oskar_Settings.h>
#include <oskar_station.h>
#include <oskar_station_work.h>
#include <oskar_SettingsBeamPattern.h>

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Generate coordinates for use in evaluating beam patterns.
 *
 * @details
 * Coordinates are generated according to the settings provided in the
 * oskar_SettingsBeamPattern structure and output as either horizontal (ENU)
 * direction cosines or phase centre relative direction cosines according to
 * the value of @p coord_type.
 *
 * @param[in]  beam_coord_type Enumerator describing the beam phase centre
 *                             coordinate type, with possible values of:
 *                             OSKAR_SPHERICAL_TYPE_EQUATORIAL
 *                             OSKAR_SPHERICAL_TYPE_AZEL
 * @param[in]  beam_lon  Longitude of the beam phase centre, in radians.
 * @param[in]  beam_lat  Longitude of the beam phase centre, in radians.
 * @param[in]  settings  Settings structure describing the specification
 *                            of coordinates to be generated.
 * @param[out]  coord_type  Coordinate type of the returned direction cosines.
 *                          Possible values are:
 *                          OSKAR_RELATIVE_DIRECTIONS or
 *                          OSKAR_ENU_DIRECTIONS
 * @param[out] lon0 Reference longitude of direction cosines.
 * @param[out] lat0 Reference latitude of direction cosines.
 * @param[out]  x  Array of x direction cosines.
 * @param[out]  y  Array of y direction cosines.
 * @param[out]  z  Array of z direction cosines.
 * @param[in,out] status Error status code.
 *
 * @return The number of pixels generated.
 */
OSKAR_APPS_EXPORT
size_t oskar_beam_pattern_generate_coordinates(int beam_coord_type,
        double beam_lon, double beam_lat, const oskar_SettingsBeamPattern* s,
        int* coord_type, double* lon0, double* lat0,
        oskar_Mem* x, oskar_Mem* y, oskar_Mem* z, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_BEAM_PATTERN_GENERATE_COORDINATES_H_ */
