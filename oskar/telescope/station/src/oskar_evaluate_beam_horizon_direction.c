/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include "telescope/station/oskar_evaluate_beam_horizon_direction.h"
#include "convert/oskar_convert_apparent_ra_dec_to_enu_directions.h"
#include "math/oskar_angular_distance.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_beam_horizon_direction(double* x, double* y, double* z,
        const oskar_Station* station, const double gast, int* status)
{
    int beam_coord_type;
    double beam_lon, beam_lat;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get direction cosines in horizontal coordinates. */
    beam_coord_type = oskar_station_beam_coord_type(station);
    beam_lon = oskar_station_beam_lon_rad(station);
    beam_lat = oskar_station_beam_lat_rad(station);

    if (beam_coord_type == OSKAR_SPHERICAL_TYPE_EQUATORIAL)
    {
        double lon, lat, last;
        lon = oskar_station_lon_rad(station);
        lat = oskar_station_lat_rad(station);
        last = gast + lon; /* Local Apparent Sidereal Time, in radians. */
        oskar_convert_apparent_ra_dec_to_enu_directions_d(1, &beam_lon,
                &beam_lat, last, lat, x, y, z);
    }
    else if (beam_coord_type == OSKAR_SPHERICAL_TYPE_AZEL)
    {
        /* Convert AZEL to direction cosines. */
        double cos_lat;
        cos_lat = cos(beam_lat);
        *x = cos_lat * sin(beam_lon);
        *y = cos_lat * cos(beam_lon);
        *z = sin(beam_lat);
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
    }

    /* Check if the beam direction needs to be one of a set of
     * allowed (az,el) directions. */
    if (oskar_station_num_permitted_beams(station) > 0)
    {
        int i, n, min_index = 0;
        double az, el, cos_el, min_dist = DBL_MAX;
        const double *p_az, *p_el;

        /* Convert current direction cosines to azimuth, elevation. */
        az = atan2(*x, *y);
        el = atan2(*z, sqrt(*x * *x + *y * *y));

        /* Get pointers to permitted beam data. */
        n = oskar_station_num_permitted_beams(station);
        p_az = oskar_mem_double_const(
                oskar_station_permitted_beam_az_rad_const(station),
                status);
        p_el = oskar_mem_double_const(
                oskar_station_permitted_beam_el_rad_const(station),
                status);

        /* Loop over permitted beams. */
        for (i = 0; i < n; ++i)
        {
            double dist;
            dist = oskar_angular_distance(p_az[i], az, p_el[i], el);
            if (dist < min_dist)
            {
                min_dist = dist;
                min_index = i;
            }
        }

        /* Select beam azimuth and elevation based on minimum distance. */
        az = p_az[min_index];
        el = p_el[min_index];

        /* Set new direction cosines. */
        cos_el = cos(el);
        *x = cos_el * sin(az);
        *y = cos_el * cos(az);
        *z = sin(el);
    }
}

#ifdef __cplusplus
}
#endif
