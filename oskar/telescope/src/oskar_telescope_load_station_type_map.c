/*
 * Copyright (c) 2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/private_telescope.h"
#include "telescope/oskar_telescope.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_load_station_type_map(oskar_Telescope* telescope,
        const char* filename, int* status)
{
    oskar_Mem* types = oskar_mem_create(OSKAR_INT, OSKAR_CPU, 0, status);
    const int num_stations = (int) oskar_mem_load_ascii(
            filename, 1, status, types, "");
    if (num_stations != telescope->num_stations)
    {
        *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE_ENTRIES_MISMATCH;
    }
    oskar_mem_copy(telescope->station_type_map, types, status);
    oskar_mem_free(types, status);
}

#ifdef __cplusplus
}
#endif
