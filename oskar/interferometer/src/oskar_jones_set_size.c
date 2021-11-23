/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "interferometer/private_jones.h"

#include "interferometer/oskar_jones_set_size.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_jones_set_size(oskar_Jones* jones, int num_stations,
        int num_sources, int* status)
{
    int capacity = 0;
    if (*status) return;

    /* Check size is within existing capacity. */
    capacity = jones->cap_stations * jones->cap_sources;
    if (num_stations * num_sources > capacity)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    /* Set the new dimension sizes, but don't actually resize the memory. */
    jones->num_stations = num_stations;
    jones->num_sources = num_sources;
}

#ifdef __cplusplus
}
#endif
