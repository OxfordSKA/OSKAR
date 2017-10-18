/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#include "telescope/station/oskar_evaluate_station_from_telescope_dipole_azimuth.h"
#include "math/oskar_cmath.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

float oskar_evaluate_station_from_telescope_dipole_azimuth_f(
        float telescope_lon_rad, float telescope_lat_rad,
        float station_lon_rad, float station_lat_rad)
{
    /* Initial: station position.
     * Final: telescope position.
     * Add (initial - final) to dipole orientations. */
    float delta_lon, sin_delta, cos_delta;
    float sin_telescope, cos_telescope, sin_station, cos_station;
    float initial_bearing, final_bearing;

    /* Pre-calculate required terms. */
    delta_lon = telescope_lon_rad - station_lon_rad;
    sin_delta = sinf(delta_lon);
    cos_delta = cosf(delta_lon);
    sin_telescope = sinf(telescope_lat_rad);
    cos_telescope = cosf(telescope_lat_rad);
    sin_station = sinf(station_lat_rad);
    cos_station = cosf(station_lat_rad);

    /* Initial bearing. */
    initial_bearing = atan2f(sin_delta * cos_telescope,
            cos_station * sin_telescope -
            sin_station * cos_telescope * cos_delta);
    initial_bearing = fmodf(initial_bearing, 2.0f * (float) M_PI);
    if (initial_bearing < 0.0f)
        initial_bearing += 2.0f * (float) M_PI;

    /* Final bearing. */
    final_bearing = atan2f(sin_delta * cos_station,
            cos_telescope * sin_station -
            sin_telescope * cos_station * cos_delta);
    final_bearing = ((float) M_PI) - fmodf(final_bearing, 2.0f * (float) M_PI);
    if (final_bearing < 0.0f)
        final_bearing += 2.0f * (float) M_PI;

    /* Return difference. */
    return initial_bearing - final_bearing;
}

double oskar_evaluate_station_from_telescope_dipole_azimuth_d(
        double telescope_lon_rad, double telescope_lat_rad,
        double station_lon_rad, double station_lat_rad)
{
    /* Initial: station position.
     * Final: telescope position.
     * Add (initial - final) to dipole orientations. */
    double delta_lon, sin_delta, cos_delta;
    double sin_telescope, cos_telescope, sin_station, cos_station;
    double initial_bearing, final_bearing;

    /* Pre-calculate required terms. */
    delta_lon = telescope_lon_rad - station_lon_rad;
    sin_delta = sin(delta_lon);
    cos_delta = cos(delta_lon);
    sin_telescope = sin(telescope_lat_rad);
    cos_telescope = cos(telescope_lat_rad);
    sin_station = sin(station_lat_rad);
    cos_station = cos(station_lat_rad);

    /* Initial bearing. */
    initial_bearing = atan2(sin_delta * cos_telescope,
            cos_station * sin_telescope -
            sin_station * cos_telescope * cos_delta);
    initial_bearing = fmod(initial_bearing, 2.0 * M_PI);
    if (initial_bearing < 0.0)
        initial_bearing += 2.0 * M_PI;

    /* Final bearing. */
    final_bearing = atan2(sin_delta * cos_station,
            cos_telescope * sin_station -
            sin_telescope * cos_station * cos_delta);
    final_bearing = M_PI - fmod(final_bearing, 2.0 * M_PI);
    if (final_bearing < 0.0)
        final_bearing += 2.0 * M_PI;

    /* Return difference. */
    return initial_bearing - final_bearing;
}

#ifdef __cplusplus
}
#endif
