/*
 * Copyright (c) 2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"

void create_sky_model(const char* filename, int* status)
{
    const int precision = OSKAR_DOUBLE;
    oskar_Sky* sky = oskar_sky_create(precision, OSKAR_CPU, 3, status);
    oskar_sky_set_source_str(sky, 0,
            "20.0 -30.0 1 0 0 0 100e6 -0.7 0   0  0   0", status);
    oskar_sky_set_source_str(sky, 1,
            "20.0 -30.5 3 2 2 0 100e6 -0.7 0 600 50  45", status);
    oskar_sky_set_source_str(sky, 2,
            "20.5 -30.5 3 0 0 2 100e6 -0.7 0 700 10 -10", status);
    oskar_sky_save(sky, filename, status);
    oskar_sky_free(sky, status);
}
