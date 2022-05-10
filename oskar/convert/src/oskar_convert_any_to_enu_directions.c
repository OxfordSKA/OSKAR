/*
 * Copyright (c) 2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_any_to_enu_directions.h"
#include "convert/oskar_convert_apparent_ra_dec_to_enu_directions.h"
#include "convert/oskar_convert_az_el_to_enu_directions.h"
#include "convert/oskar_convert_relative_directions_to_enu_directions.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_any_to_enu_directions(
        int coord_type,
        int num_points,
        const oskar_Mem* const coords[3],
        double ref_lon_rad,
        double ref_lat_rad,
        double lst_rad,
        double lat_rad,
        oskar_Mem* enu[3],
        int* status)
{
    if (*status) return;
    switch (coord_type)
    {
    case OSKAR_COORDS_REL_DIR:
        oskar_convert_relative_directions_to_enu_directions(
                0, 0, 0, num_points,
                coords[0], coords[1], coords[2],
                (lst_rad - ref_lon_rad), ref_lat_rad, lat_rad, 0,
                enu[0], enu[1], enu[2], status);
        break;
    case OSKAR_COORDS_ENU_DIR:
        oskar_mem_copy_contents(enu[0], coords[0], 0, 0,
                (size_t)num_points, status);
        oskar_mem_copy_contents(enu[1], coords[1], 0, 0,
                (size_t)num_points, status);
        oskar_mem_copy_contents(enu[2], coords[2], 0, 0,
                (size_t)num_points, status);
        break;
    case OSKAR_COORDS_RADEC:
        oskar_convert_apparent_ra_dec_to_enu_directions(num_points,
                coords[0], coords[1], lst_rad, lat_rad, 0,
                enu[0], enu[1], enu[2], status);
        break;
    case OSKAR_COORDS_AZEL:
        oskar_convert_az_el_to_enu_directions(num_points,
                coords[0], coords[1], enu[0], enu[1], enu[2], status);
        break;
    default:
        *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
        return;
    }
}

#ifdef __cplusplus
}
#endif
