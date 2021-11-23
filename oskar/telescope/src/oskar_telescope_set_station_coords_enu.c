/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_ecef_to_geodetic_spherical.h"
#include "convert/oskar_convert_enu_to_offset_ecef.h"
#include "convert/oskar_convert_offset_ecef_to_ecef.h"
#include "telescope/oskar_telescope.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_set_station_coords_enu(oskar_Telescope* telescope,
        double longitude_rad, double latitude_rad, double altitude_m,
        int num_stations, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, const oskar_Mem* x_err, const oskar_Mem* y_err,
        const oskar_Mem* z_err, int* status)
{
    int i = 0;

    /* Check lengths. */
    if ((int)oskar_mem_length(x) < num_stations ||
            (int)oskar_mem_length(y) < num_stations ||
            (int)oskar_mem_length(z) < num_stations ||
            (int)oskar_mem_length(x_err) < num_stations ||
            (int)oskar_mem_length(y_err) < num_stations ||
            (int)oskar_mem_length(z_err) < num_stations)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Resize the telescope model to hold the station data. */
    /* Do not resize the station array here. */
    oskar_telescope_resize(telescope, num_stations, status);
    if (*status) return;

    /* Store the telescope centre longitude, latitude, and altitude. */
    oskar_telescope_set_position(telescope,
            longitude_rad, latitude_rad, altitude_m);

    /* Loop over station coordinates. */
    for (i = 0; i < num_stations; ++i)
    {
        double true_geodetic[3]; /* Longitude, latitude, altitude. */

        /* x, y, z, delta x, delta y, delta z */
        double hor[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double ecef[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double offset_ecef[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        if (*status) break;

        /* Get coordinates from input arrays. */
        hor[0] = oskar_mem_get_element(x, i, status);
        hor[1] = oskar_mem_get_element(y, i, status);
        hor[2] = oskar_mem_get_element(z, i, status);
        hor[3] = oskar_mem_get_element(x_err, i, status);
        hor[4] = oskar_mem_get_element(y_err, i, status);
        hor[5] = oskar_mem_get_element(z_err, i, status);

        /* Get "true" coordinates ([3, 4, 5]) from "measured" coordinates. */
        hor[3] += hor[0];
        hor[4] += hor[1];
        hor[5] += hor[2];

        /* Convert horizon plane to offset geocentric Cartesian coordinates. */
        oskar_convert_enu_to_offset_ecef_d(1, &hor[0], &hor[1], &hor[2],
                longitude_rad, latitude_rad,
                &offset_ecef[0], &offset_ecef[1], &offset_ecef[2]);
        oskar_convert_enu_to_offset_ecef_d(1, &hor[3], &hor[4], &hor[5],
                longitude_rad, latitude_rad,
                &offset_ecef[3], &offset_ecef[4], &offset_ecef[5]);

        /* Convert to ECEF, then to station longitude, latitude, altitude. */
        oskar_convert_offset_ecef_to_ecef(
                1, &offset_ecef[3], &offset_ecef[4], &offset_ecef[5],
                longitude_rad, latitude_rad, altitude_m,
                &ecef[3], &ecef[4], &ecef[5]);
        oskar_convert_ecef_to_geodetic_spherical(1,
                &ecef[3], &ecef[4], &ecef[5],
                &true_geodetic[0], &true_geodetic[1], &true_geodetic[2]);

        /* Store the coordinates. */
        oskar_telescope_set_station_coords(telescope, i, true_geodetic,
                &offset_ecef[0], &offset_ecef[3], &hor[0], &hor[3], status);
    }
}

#ifdef __cplusplus
}
#endif
