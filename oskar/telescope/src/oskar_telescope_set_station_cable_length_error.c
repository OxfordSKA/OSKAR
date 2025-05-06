/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>

#include "telescope/private_telescope.h"
#include "telescope/oskar_telescope.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_set_station_cable_length_error(
        oskar_Telescope* telescope,
        int feed,
        int index,
        const double error_metres,
        int* status
)
{
    oskar_Mem* ptr = 0;
    if (*status || !telescope) return;
    if (index >= telescope->num_stations || feed > 1)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }
    const int type = telescope->precision;
    const int loc = telescope->mem_location;
    ptr = telescope->station_cable_length_error[feed];
    if (!ptr)
    {
        telescope->station_cable_length_error[feed] = oskar_mem_create(
                type, loc, telescope->num_stations, status
        );
        ptr = telescope->station_cable_length_error[feed];
    }
    if (loc == OSKAR_CPU)
    {
        void *cable = oskar_mem_void(ptr);
        if (type == OSKAR_DOUBLE)
        {
            ((double*)cable)[index] = error_metres;
        }
        else if (type == OSKAR_SINGLE)
        {
            ((float*)cable)[index] = (float)error_metres;
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else
    {
        oskar_mem_set_element_real(ptr, index, error_metres, status);
    }
}

#ifdef __cplusplus
}
#endif
