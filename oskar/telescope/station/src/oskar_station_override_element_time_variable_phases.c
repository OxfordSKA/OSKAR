/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_override_element_time_variable_phases(
        oskar_Station* station, int feed, double phase_std_rad, int* status)
{
    int i = 0;
    if (*status || !station) return;

    /* Override element data only at last level. */
    if (oskar_station_has_child(station))
    {
        for (i = 0; i < station->num_elements; ++i)
        {
            oskar_station_override_element_time_variable_phases(
                    oskar_station_child(station, i),
                    feed, phase_std_rad, status);
        }
    }
    else
    {
        oskar_Mem* ptr = station->element_phase_error_rad[feed];
        if (!ptr)
        {
            station->element_phase_error_rad[feed] = oskar_mem_create(
                    station->precision, station->mem_location,
                    station->num_elements, status);
            ptr = station->element_phase_error_rad[feed];
        }
        oskar_mem_set_value_real(ptr,
                phase_std_rad, 0, station->num_elements, status);
    }
}

#ifdef __cplusplus
}
#endif
