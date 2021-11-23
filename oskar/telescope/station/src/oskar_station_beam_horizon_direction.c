/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/oskar_station.h"
#include "convert/oskar_convert_apparent_ra_dec_to_enu_directions.h"
#include "math/oskar_angular_distance.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_beam_horizon_direction(const oskar_Station* station,
        const double gast_rad, double* x, double* y, double* z, int* status)
{
    if (*status) return;

    /* Get direction cosines in horizontal coordinates. */
    const int beam_coord_type = oskar_station_beam_coord_type(station);
    const double beam_lon_rad = oskar_station_beam_lon_rad(station);
    const double beam_lat_rad = oskar_station_beam_lat_rad(station);
    if (beam_coord_type == OSKAR_COORDS_RADEC)
    {
        const double last_rad = gast_rad +
                oskar_station_lon_rad(station); /* Local Sidereal Time. */
        const double sin_lat = sin(oskar_station_lat_rad(station));
        const double cos_lat = cos(oskar_station_lat_rad(station));
        oskar_convert_apparent_ra_dec_to_enu_directions_double(1, &beam_lon_rad,
                &beam_lat_rad, last_rad, sin_lat, cos_lat, 0, x, y, z);
    }
    else if (beam_coord_type == OSKAR_COORDS_AZEL)
    {
        /* Convert AZEL to direction cosines. */
        const double cos_lat = cos(beam_lat_rad);
        *x = cos_lat * sin(beam_lon_rad);
        *y = cos_lat * cos(beam_lon_rad);
        *z = sin(beam_lat_rad);
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
    }

    /* Check if the beam direction needs to be one of a set of
     * allowed (az,el) directions. */
    if (oskar_station_num_permitted_beams(station) > 0)
    {
        int i = 0, n = 0, min_index = 0;
        double az = 0.0, el = 0.0, cos_el = 0.0, min_dist = DBL_MAX;
        const double *p_az = 0, *p_el = 0;

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
            const double d = oskar_angular_distance(p_az[i], az, p_el[i], el);
            if (d < min_dist)
            {
                min_dist = d;
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
