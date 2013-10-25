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

#include <oskar_evaluate_beam_horizon_direction.h>
#include <oskar_convert_apparent_ra_dec_to_horizon_direction.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_beam_horizon_direction(double* x, double* y, double* z,
        const oskar_Station* station, const double gast, int* status)
{
    int beam_coord_type;
    double beam_ra, beam_dec;

    /* Check all inputs. */
    if (!x || !y || !z || !station || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Convert equatorial to horizontal coordinates if necessary. */
    beam_coord_type = oskar_station_beam_coord_type(station);
    beam_ra = oskar_station_beam_longitude_rad(station);
    beam_dec = oskar_station_beam_latitude_rad(station);

    if (beam_coord_type == OSKAR_SPHERICAL_TYPE_EQUATORIAL)
    {
        double lon, lat, last;
        lon = oskar_station_longitude_rad(station);
        lat = oskar_station_latitude_rad(station);
        last = gast + lon; /* Local Apparent Sidereal Time, in radians. */
        oskar_convert_apparent_ra_dec_to_horizon_direction_d(1, &beam_ra,
                &beam_dec, last, lat, x, y, z);
    }
    else if (beam_coord_type == OSKAR_SPHERICAL_TYPE_HORIZONTAL)
    {
        /* Convert AZEL to direction cosines. */
        double cos_lat;
        cos_lat = cos(beam_dec);
        *x = cos_lat * sin(beam_ra);
        *y = cos_lat * cos(beam_ra);
        *z = sin(beam_dec);
    }
    else
    {
        *status = OSKAR_ERR_SETTINGS;
    }
}

#ifdef __cplusplus
}
#endif
