/*
 * Copyright (c) 2013-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>

#include "sky/private_sky.h"
#include "sky/oskar_sky.h"

#ifdef __cplusplus
extern "C" {
#endif


oskar_Sky* oskar_sky_create(
        int type,
        int location,
        int num_sources,
        int* status
)
{
    int capacity = 0;
    oskar_Sky* model = 0;
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;                /* LCOV_EXCL_LINE */
        return 0;                                         /* LCOV_EXCL_LINE */
    }
    model = (oskar_Sky*) calloc(1, sizeof(oskar_Sky));
    if (!model)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;         /* LCOV_EXCL_LINE */
        return 0;                                         /* LCOV_EXCL_LINE */
    }
    const int factor = 256;
    capacity = num_sources + 1; /* Add 1 for normalisation source. */
    /* Round up capacity to next highest multiple.
     * Needed for sub-buffer column alignment. */
    capacity = capacity + (factor - capacity % factor) % factor;
    model->attr_int[OSKAR_SKY_PRECISION] = type;
    model->attr_int[OSKAR_SKY_MEM_LOCATION] = location;
    model->attr_int[OSKAR_SKY_CAPACITY] = capacity;
    model->attr_int[OSKAR_SKY_NUM_SOURCES] = num_sources;
    model->table = oskar_mem_create(type, location, 0, status);
    return model;
}

#ifdef __cplusplus
}
#endif
