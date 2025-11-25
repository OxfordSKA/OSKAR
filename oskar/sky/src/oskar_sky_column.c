/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/private_sky.h"
#include "sky/oskar_sky.h"

#ifdef __cplusplus
extern "C" {
#endif


oskar_Mem* oskar_sky_column(
        oskar_Sky* sky,
        oskar_SkyColumn column_type,
        int column_attribute,
        int* status
)
{
    int i = 0, num_columns = 0;
    if (*status || column_type < 0) return 0;

    /* Scan the columns to return the specified type. */
    num_columns = sky->attr_int[OSKAR_SKY_NUM_COLUMNS];
    for (i = 0; i < num_columns; ++i)
    {
        if (sky->column_type[i] == column_type &&
                sky->column_attr[i] == column_attribute)
        {
            return sky->columns[i];
        }
    }

    /* The column was not found, so create it. */
    num_columns = ++(sky->attr_int[OSKAR_SKY_NUM_COLUMNS]);
    sky->column_type = (oskar_SkyColumn*) realloc(
            sky->column_type, num_columns * sizeof(oskar_SkyColumn)
    );
    sky->column_attr = (int*) realloc(
            sky->column_attr, num_columns * sizeof(int)
    );
    sky->columns = (oskar_Mem**) realloc(
            sky->columns, num_columns * sizeof(oskar_Mem*)
    );
    if (!sky->column_type || !sky->column_attr || !sky->columns)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;         /* LCOV_EXCL_LINE */
        return 0;                                         /* LCOV_EXCL_LINE */
    }
    sky->column_type[i] = column_type;
    sky->column_attr[i] = column_attribute;
    const int precision = sky->attr_int[OSKAR_SKY_PRECISION];
    const int location = sky->attr_int[OSKAR_SKY_MEM_LOCATION];
    const int capacity = sky->attr_int[OSKAR_SKY_CAPACITY];
    sky->columns[i] = oskar_mem_create(precision, location, capacity, status);
    oskar_mem_realloc(sky->ptr_columns, num_columns, status);
    oskar_mem_set_element_ptr(
            sky->ptr_columns, i, oskar_mem_void(sky->columns[i]), status
    );
    return sky->columns[i];
}


const oskar_Mem* oskar_sky_column_const(
        const oskar_Sky* sky,
        oskar_SkyColumn column_type,
        int column_attribute
)
{
    int i = 0;
    if (column_type < 0) return 0;

    /* Scan the columns to return the specified type. */
    const int num_columns = sky->attr_int[OSKAR_SKY_NUM_COLUMNS];
    for (i = 0; i < num_columns; ++i)
    {
        if (sky->column_type[i] == column_type &&
                sky->column_attr[i] == column_attribute)
        {
            return sky->columns[i];
        }
    }

    /* Column does not exist, so return NULL. */
    return 0;
}

#ifdef __cplusplus
}
#endif
