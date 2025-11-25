/*
 * Copyright (c) 2013-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"

#ifdef __cplusplus
extern "C" {
#endif


oskar_Sky* oskar_sky_create_copy(
        const oskar_Sky* src,
        int location,
        int* status
)
{
    oskar_Sky* model = 0;
    if (*status) return model;
    model = oskar_sky_create(
            oskar_sky_int(src, OSKAR_SKY_PRECISION),
            location,
            oskar_sky_int(src, OSKAR_SKY_NUM_SOURCES),
            status
    );
    oskar_sky_copy(model, src, status);
    return model;
}

#ifdef __cplusplus
}
#endif
