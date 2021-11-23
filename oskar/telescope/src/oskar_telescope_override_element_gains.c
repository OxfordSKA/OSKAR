/*
 * Copyright (c) 2019-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/oskar_telescope.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_override_element_gains(oskar_Telescope* t, int feed,
        unsigned int seed, double gain_mean, double gain_std, int* status)
{
    int i = 0;
    const int num_stations = oskar_telescope_num_station_models(t);
    for (i = 0; i < num_stations; ++i)
    {
        oskar_station_override_element_gains(oskar_telescope_station(t, i),
                feed, seed, gain_mean, gain_std, status);
    }
}

#ifdef __cplusplus
}
#endif
