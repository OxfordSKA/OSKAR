/*
 * Copyright (c) 2019-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"
#include "math/oskar_random_gaussian.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_override_element_cable_length_errors(oskar_Station* station,
        int feed, unsigned int seed, double mean_metres, double std_metres,
        int* status)
{
    int i = 0;
    if (*status || !station) return;
    if (oskar_station_mem_location(station) != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }
    const int num = station->num_elements;
    if (oskar_station_has_child(station))
    {
        /* Recursive call to find the last level (i.e. the element data). */
        for (i = 0; i < num; ++i)
        {
            oskar_station_override_element_cable_length_errors(
                    oskar_station_child(station, i),
                    feed, seed, mean_metres, std_metres, status);
        }
    }
    else
    {
        /* Override element data at last level. */
        oskar_Mem* ptr = 0;
        double r[2];
        const int type = oskar_station_precision(station);
        const int id = oskar_station_unique_id(station);
        ptr = station->element_cable_length_error[feed];
        if (!ptr)
        {
            station->element_cable_length_error[feed] = oskar_mem_create(
                    type, station->mem_location, num, status);
            ptr = station->element_cable_length_error[feed];
        }
        if (type == OSKAR_DOUBLE)
        {
            double* cable = oskar_mem_double(ptr, status);
            for (i = 0; i < num; ++i)
            {
                oskar_random_gaussian2(seed, i, id, r);
                cable[i] = mean_metres + std_metres * r[0];
            }
        }
        else if (type == OSKAR_SINGLE)
        {
            float* cable = oskar_mem_float(ptr, status);
            for (i = 0; i < num; ++i)
            {
                oskar_random_gaussian2(seed, i, id, r);
                cable[i] = mean_metres + std_metres * r[0];
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
