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
    size_t i = 0, num_columns = 0;
    if (*status || column_type < 0) return 0;

    /* Scan the columns to return the specified type. */
    num_columns = (size_t) sky->attr_int[OSKAR_SKY_NUM_COLUMNS];
    for (i = 0; i < num_columns; ++i)
    {
        if (sky->column_type[i] == column_type &&
                sky->column_attr[i] == column_attribute)
        {
            return sky->columns[i];
        }
    }

    /* The column was not found, so create it. */
    num_columns = (size_t) (++(sky->attr_int[OSKAR_SKY_NUM_COLUMNS]));
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
    const size_t capacity = (size_t) (sky->attr_int[OSKAR_SKY_CAPACITY]);
    oskar_mem_realloc(sky->table, capacity * num_columns, status);
    sky->column_type[num_columns - 1] = column_type;
    sky->column_attr[num_columns - 1] = column_attribute;
    sky->columns[num_columns - 1] = oskar_mem_create_alias(0, 0, 0, status);

    /* Need to refresh all the cached aliases because of the reallocation. */
    for (i = 0; i < num_columns; ++i)
    {
        oskar_mem_set_alias(
                sky->columns[i], sky->table, i * capacity, capacity, status
        );
    }
    oskar_mem_clear_contents(sky->columns[num_columns - 1], status);
    return sky->columns[num_columns - 1];
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
