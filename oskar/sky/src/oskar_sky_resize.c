/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/private_sky.h"
#include "sky/oskar_sky.h"

#ifdef __cplusplus
extern "C" {
#endif


void oskar_sky_resize(oskar_Sky* sky, int num_sources, int* status)
{
    int i = 0;
    if (*status || sky->attr_int[OSKAR_SKY_NUM_SOURCES] == num_sources) return;
    const int capacity = num_sources + 1;
    const int num_columns = sky->attr_int[OSKAR_SKY_NUM_COLUMNS];
    for (i = 0; i < num_columns; ++i)
    {
        oskar_mem_realloc(sky->columns[i], capacity, status);
        oskar_mem_set_element_ptr(
                sky->ptr_columns, i,
                oskar_mem_void(sky->columns[i]), status
        );
    }
    sky->attr_int[OSKAR_SKY_CAPACITY] = capacity;
    sky->attr_int[OSKAR_SKY_NUM_SOURCES] = num_sources;
}

#ifdef __cplusplus
}
#endif
