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

#ifndef OSKAR_EVALUATE_STATION_BEAM_H_
#define OSKAR_EVALUATE_STATION_BEAM_H_

/**
 * @file oskar_evaluate_station_beam.h
 */

#include <oskar_global.h>
#include <oskar_station.h>
#include <oskar_station_work.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluate the beam pattern for a station.
 *
 * @details
 * This top-level function evaluates the beam pattern of a station at the
 * specified positions, given as direction cosines.
 *
 * The station counter must be unique for the given time index.
 * It is updated automatically on exit.
 *
 * The longitude and latitude of the phase centre are used only
 * when performing beam normalisation for ENU directions.
 *
 * @param[out] beam_pattern   Output beam pattern data.
 * @param[in] num_points      Number of direction cosines given.
 * @param[in] x               Direction cosines (x direction).
 * @param[in] y               Direction cosines (y direction).
 * @param[in] z               Direction cosines (z direction).
 * @param[in] coord_type      Type of direction cosines
 *                            (OSKAR_RELATIVE_DIRECTIONS or
 *                            OSKAR_ENU_DIRECTIONS).
 * @param[in] lon0_rad        Longitude of phase centre, in radians.
 * @param[in] lat0_rad        Latitude of phase centre, in radians.
 * @param[in] station         Station model.
 * @param[in] work            Station beam work arrays.
 * @param[in] time_index      Simulation time index.
 * @param[in,out] station_counter Station counter. Must be unique for given time.
 * @param[in] frequency       The observing frequency in Hz.
 * @param[in] gast            The Greenwich Apparent Sidereal Time, in radians.
 * @param[in,out] status      Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_station_beam(oskar_Mem* beam_pattern, int num_points,
        oskar_Mem* x, oskar_Mem* y, oskar_Mem* z, int coord_type,
        double lon0_rad, double lat0_rad, const oskar_Station* station,
        oskar_StationWork* work, int time_index, int* station_counter,
        double frequency, double gast, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_STATION_BEAM_H_ */
