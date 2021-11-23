/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_override_element_time_variable_gains(
        oskar_Station* station, int feed, double gain_std, int* status)
{
    int i = 0;
    if (*status || !station) return;

    /* Override element data only at last level. */
    if (oskar_station_has_child(station))
    {
        for (i = 0; i < station->num_elements; ++i)
        {
            oskar_station_override_element_time_variable_gains(
                    oskar_station_child(station, i), feed, gain_std, status);
        }
    }
    else
    {
        oskar_Mem* ptr = station->element_gain_error[feed];
        if (!ptr)
        {
            station->element_gain_error[feed] = oskar_mem_create(
                    station->precision, station->mem_location,
                    station->num_elements, status);
            ptr = station->element_gain_error[feed];
        }
        oskar_mem_set_value_real(ptr,
                gain_std, 0, station->num_elements, status);
    }
}

#ifdef __cplusplus
}
#endif
