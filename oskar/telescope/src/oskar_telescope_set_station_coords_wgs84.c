/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_cmath.h"
#include "convert/oskar_convert_ecef_to_enu.h"
#include "convert/oskar_convert_enu_to_offset_ecef.h"
#include "convert/oskar_convert_geodetic_spherical_to_ecef.h"
#include "telescope/oskar_telescope.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_set_station_coords_wgs84(oskar_Telescope* telescope,
        double longitude_rad, double latitude_rad, double altitude_m,
        int num_stations, const oskar_Mem* lon_deg, const oskar_Mem* lat_deg,
        const oskar_Mem* alt_m, int* status)
{
    int i = 0;

    /* Check lengths. */
    if ((int)oskar_mem_length(lon_deg) < num_stations ||
            (int)oskar_mem_length(lat_deg) < num_stations ||
            (int)oskar_mem_length(alt_m) < num_stations)
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
        /* longitude, latitude, altitude */
        double wgs[] = {0.0, 0.0, 0.0};
        double ecef[] = {0.0, 0.0, 0.0};
        double offset_ecef[] = {0.0, 0.0, 0.0};
        double hor[] = {0.0, 0.0, 0.0};
        if (*status) break;

        /* Get coordinates from input arrays. */
        wgs[0] = oskar_mem_get_element(lon_deg, i, status);
        wgs[1] = oskar_mem_get_element(lat_deg, i, status);
        wgs[2] = oskar_mem_get_element(alt_m, i, status);

        /* Convert geodetic spherical to ECEF. */
        wgs[0] *= M_PI / 180.0;
        wgs[1] *= M_PI / 180.0;
        oskar_convert_geodetic_spherical_to_ecef(1, &wgs[0], &wgs[1], &wgs[2],
                &ecef[0], &ecef[1], &ecef[2]);

        /* Convert station ECEF to horizon plane coordinates. */
        oskar_convert_ecef_to_enu(1, &ecef[0], &ecef[1], &ecef[2],
                longitude_rad, latitude_rad, altitude_m, &hor[0], &hor[1], &hor[2]);
        oskar_convert_enu_to_offset_ecef_d(1, &hor[0], &hor[1], &hor[2],
                longitude_rad, latitude_rad,
                &offset_ecef[0], &offset_ecef[1], &offset_ecef[2]);

        /* Store the coordinates. */
        oskar_telescope_set_station_coords(telescope, i, wgs,
                &ecef[0], &ecef[0], &hor[0], &hor[0], status);
    }
}

#ifdef __cplusplus
}
#endif
