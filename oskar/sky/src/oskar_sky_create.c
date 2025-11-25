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
    model->attr_double = (double*) calloc(
            OSKAR_SKY_NUM_ATTRIBUTES_DOUBLE, sizeof(double)
    );
    model->attr_int = (int*) calloc(OSKAR_SKY_NUM_ATTRIBUTES_INT, sizeof(int));
    model->attr_int[OSKAR_SKY_PRECISION] = type;
    model->attr_int[OSKAR_SKY_MEM_LOCATION] = location;
    model->attr_int[OSKAR_SKY_CAPACITY] = num_sources + 1;
    model->attr_int[OSKAR_SKY_NUM_SOURCES] = num_sources;
    model->ptr_columns = oskar_mem_create(OSKAR_PTR, location, 0, status);
    return model;
}

#ifdef __cplusplus
}
#endif
