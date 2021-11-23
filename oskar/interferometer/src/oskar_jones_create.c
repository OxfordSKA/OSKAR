/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "interferometer/private_jones.h"
#include "interferometer/oskar_jones.h"
#include "mem/oskar_mem.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Jones* oskar_jones_create(int type, int location, int num_stations,
        int num_sources, int* status)
{
    oskar_Jones* jones = 0;
    const int base_type = oskar_type_precision(type);
    if (!oskar_type_is_complex(type) ||
            (base_type != OSKAR_SINGLE && base_type != OSKAR_DOUBLE))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }
    jones = (oskar_Jones*) calloc(1, sizeof(oskar_Jones));
    jones->num_stations = num_stations;
    jones->num_sources = num_sources;
    jones->cap_stations = num_stations;
    jones->cap_sources = num_sources;
    jones->data = oskar_mem_create(type, location,
            num_stations * num_sources, status);
    return jones;
}

#ifdef __cplusplus
}
#endif
