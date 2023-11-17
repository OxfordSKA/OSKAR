/*
 * Copyright (c) 2023, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/oskar_station.h"
#include "telescope/station/private_station.h"
#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_load_virtual_antenna_angle(oskar_Station* station,
        const char* filename, int* status)
{
    oskar_Mem* angles = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    const int num_stations = (int) oskar_mem_load_ascii(
            filename, 1, status, angles, "");
    if (num_stations != 1)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    const double* angles_ptr = oskar_mem_double(angles, status);
    oskar_station_set_virtual_antenna_angle(station, angles_ptr[0]);
    oskar_mem_free(angles, status);
}

#ifdef __cplusplus
}
#endif
