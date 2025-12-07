/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"
#include "sky/private_sky.h"

#ifdef __cplusplus
extern "C" {
#endif


void oskar_sky_create_columns(
        oskar_Sky* sky,
        const oskar_Sky* src,
        int* status
)
{
    int i = 0;
    int num_columns = 0;
    if (*status) return;

    /* Clear any existing column data. */
    num_columns = sky->attr_int[OSKAR_SKY_NUM_COLUMNS];
    for (i = 0; i < num_columns; ++i)
    {
        oskar_mem_free(sky->columns[i], status);
    }
    oskar_mem_free(sky->table, status);
    free(sky->columns);
    free(sky->column_attr);
    free(sky->column_type);

    /* Find out the number of columns to create. */
    const int num_columns_src = src->attr_int[OSKAR_SKY_NUM_COLUMNS];
    const int num_columns_scratch = (
            OSKAR_SKY_SCRATCH_END - OSKAR_SKY_SCRATCH_START
    );
    num_columns = num_columns_src + num_columns_scratch;

    /* Create the new (empty) table. */
    const size_t capacity = (size_t) sky->attr_int[OSKAR_SKY_CAPACITY];
    sky->table = oskar_mem_create(
            sky->attr_int[OSKAR_SKY_PRECISION],
            sky->attr_int[OSKAR_SKY_MEM_LOCATION],
            (size_t) num_columns * capacity, status
    );
    oskar_mem_clear_contents(sky->table, status);

    /* Create arrays to store the column metadata. */
    sky->column_type = (oskar_SkyColumn*) calloc(
            num_columns, sizeof(oskar_SkyColumn)
    );
    sky->column_attr = (int*) calloc(num_columns, sizeof(int));
    sky->columns = (oskar_Mem**) calloc(num_columns, sizeof(oskar_Mem*));
    if (!sky->column_type || !sky->column_attr || !sky->columns)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;         /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }

    /* Copy the column metadata from the existing sky model. */
    for (i = 0; i < num_columns_src; ++i)
    {
        sky->column_type[i] = src->column_type[i];
        sky->column_attr[i] = src->column_attr[i];
    }

    /* Set the metadata for the scratch columns. */
    for (i = 0; i < num_columns_scratch; ++i)
    {
        sky->column_type[i + num_columns_src] = (oskar_SkyColumn) (
                i + OSKAR_SKY_SCRATCH_START
        );
    }

    /* Set all the column references. */
    for (i = 0; i < num_columns; ++i)
    {
        sky->columns[i] = oskar_mem_create_alias(
                sky->table, i * capacity, capacity, status
        );
    }

    /* Update the new number of columns. */
    sky->attr_int[OSKAR_SKY_NUM_COLUMNS] = num_columns;
}

#ifdef __cplusplus
}
#endif
