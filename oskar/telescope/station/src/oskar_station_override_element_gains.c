/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"
#include "math/oskar_random_gaussian.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_override_element_gains(oskar_Station* station, int feed,
        unsigned int seed, double gain_mean, double gain_std, int* status)
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
            oskar_station_override_element_gains(
                    oskar_station_child(station, i),
                    feed, seed, gain_mean, gain_std, status);
        }
    }
    else
    {
        /* Override element data at last level. */
        oskar_Mem* ptr = 0;
        double r[2];
        const int type = oskar_station_precision(station);
        const int id = oskar_station_unique_id(station);
        ptr = station->element_gain[feed];
        if (!ptr)
        {
            station->element_gain[feed] = oskar_mem_create(
                    type, station->mem_location, num, status);
            ptr = station->element_gain[feed];
        }
        if (gain_mean <= 0.0) gain_mean = 1.0;
        if (type == OSKAR_DOUBLE)
        {
            double* gain = oskar_mem_double(ptr, status);
            for (i = 0; i < num; ++i)
            {
                oskar_random_gaussian2(seed, i, id, r);
                gain[i] = gain_mean + gain_std * r[0];
            }
        }
        else if (type == OSKAR_SINGLE)
        {
            float* gain = oskar_mem_float(ptr, status);
            for (i = 0; i < num; ++i)
            {
                oskar_random_gaussian2(seed, i, id, r);
                gain[i] = gain_mean + gain_std * r[0];
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
