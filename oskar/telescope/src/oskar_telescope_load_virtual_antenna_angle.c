/*
 * Copyright (c) 2023, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/oskar_telescope.h"
#include "telescope/private_telescope.h"
#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_load_virtual_antenna_angle(oskar_Telescope* telescope,
        const char* filename, int* status)
{
    oskar_Mem* angles = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    const int num_stations = (int) oskar_mem_load_ascii(
            filename, 1, status, angles, "");
    if (num_stations != telescope->num_stations)
    {
        *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE_ENTRIES_MISMATCH;
    }
    else
    {
        const double* angles_ptr = oskar_mem_double(angles, status);
        for (int i = 0; i < num_stations; ++i)
        {
            oskar_Station* station = oskar_telescope_station(telescope, i);
            oskar_station_set_virtual_antenna_angle(station, angles_ptr[i]);
        }
    }
    oskar_mem_free(angles, status);
}

#ifdef __cplusplus
}
#endif
