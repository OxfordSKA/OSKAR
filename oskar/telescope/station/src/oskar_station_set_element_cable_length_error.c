/*
 * Copyright (c) 2019-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_set_element_cable_length_error(oskar_Station* station,
        int feed, int index, const double error_metres, int* status)
{
    oskar_Mem* ptr = 0;
    if (*status || !station) return;
    if (index >= station->num_elements || feed > 1)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }
    const int type = station->precision;
    const int loc = station->mem_location;
    ptr = station->element_cable_length_error[feed];
    if (!ptr)
    {
        station->element_cable_length_error[feed] = oskar_mem_create(
                type, loc, station->num_elements, status);
        ptr = station->element_cable_length_error[feed];
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
