/*
 * Copyright (c) 2013-2019, The University of Oxford
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
    oskar_Mem *lon_cpu = 0, *lat_cpu = 0, *lon_p, *lat_p;
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
            oskar_convert_relative_directions_to_lon_lat_2d_f(num_pixels,
                    oskar_mem_float_const(lon_p, status),
                    oskar_mem_float_const(lat_p, status),
                    lon_rad, lat_rad,
                    oskar_mem_float(lon_p, status),
                    oskar_mem_float(lat_p, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_evaluate_image_lm_grid_d(num_pixels_l, num_pixels_m,
                    fov_rad_lon, fov_rad_lat, oskar_mem_double(lon_p, status),
                    oskar_mem_double(lat_p, status));
            oskar_convert_relative_directions_to_lon_lat_2d_d(num_pixels,
                    oskar_mem_double_const(lon_p, status),
                    oskar_mem_double_const(lat_p, status),
                    lon_rad, lat_rad,
                    oskar_mem_double(lon_p, status),
                    oskar_mem_double(lat_p, status));
        }
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
