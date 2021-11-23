/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_relative_directions_to_lon_lat.h"
#include "math/oskar_evaluate_image_lm_grid.h"
#include "math/oskar_evaluate_image_lon_lat_grid.h"
#include "mem/oskar_mem.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_image_lon_lat_grid(int num_pixels_l, int num_pixels_m,
        double fov_rad_lon, double fov_rad_lat, double lon_rad, double lat_rad,
        oskar_Mem* lon, oskar_Mem* lat, int* status)
{
    oskar_Mem *lon_cpu = 0, *lat_cpu = 0, *lon_p = 0, *lat_p = 0;
    const int type = oskar_mem_type(lon);
    const int location = oskar_mem_location(lon);
    const int num_pixels = num_pixels_l * num_pixels_m;
    oskar_mem_ensure(lon, num_pixels, status);
    oskar_mem_ensure(lat, num_pixels, status);
    if (*status) return;
    if (location != OSKAR_CPU)
    {
        lon_cpu = oskar_mem_create(type, OSKAR_CPU, num_pixels, status);
        lat_cpu = oskar_mem_create(type, OSKAR_CPU, num_pixels, status);
        lon_p = lon_cpu;
        lat_p = lat_cpu;
    }
    else
    {
        lon_p = lon;
        lat_p = lat;
    }
    if (! *status)
    {
        if (type == OSKAR_SINGLE)
        {
            oskar_evaluate_image_lm_grid_f(num_pixels_l, num_pixels_m,
                    fov_rad_lon, fov_rad_lat, oskar_mem_float(lon_p, status),
                    oskar_mem_float(lat_p, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_evaluate_image_lm_grid_d(num_pixels_l, num_pixels_m,
                    fov_rad_lon, fov_rad_lat, oskar_mem_double(lon_p, status),
                    oskar_mem_double(lat_p, status));
        }
        oskar_convert_relative_directions_to_lon_lat(num_pixels,
                lon_p, lat_p, 0, lon_rad, lat_rad, lon_p, lat_p, status);
    }
    if (location != OSKAR_CPU)
    {
        oskar_mem_copy(lon, lon_p, status);
        oskar_mem_copy(lat, lat_p, status);
    }
    oskar_mem_free(lon_cpu, status);
    oskar_mem_free(lat_cpu, status);
}

#ifdef __cplusplus
}
#endif
