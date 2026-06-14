/*
 * Copyright (c) 2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>

#include "log/oskar_log.h"
#include "sky/oskar_sky.h"
#include "sky/private_sky.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct
{
    oskar_SkyColumn column_type;
    int column_attr;
    int column_index; /* Original column index. */
}
oskar_SkyColumnMetadata;


static int compare_column_metadata(const void* ptr_a, const void* ptr_b)
{
    const oskar_SkyColumnMetadata* a = (const oskar_SkyColumnMetadata*) ptr_a;
    const oskar_SkyColumnMetadata* b = (const oskar_SkyColumnMetadata*) ptr_b;
    if (a->column_type != b->column_type)
    {
        return ((int) a->column_type < (int) b->column_type) ? -1 : 1;
    }
    if (a->column_attr != b->column_attr)
    {
        return (a->column_attr < b->column_attr) ? -1 : 1;
    }
    return 0;
}


void oskar_sky_sort_columns(oskar_Sky* sky, int* status)
{
    int c = 0;
    if (*status || !sky) return;
    const int num_columns = sky->attr_int[OSKAR_SKY_NUM_COLUMNS];

    /* Sort the column metadata. */
    oskar_SkyColumnMetadata* metadata = (oskar_SkyColumnMetadata*) malloc(
            num_columns * sizeof(oskar_SkyColumnMetadata)
    );
    if (!metadata)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;         /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }
    for (c = 0; c < num_columns; ++c)
    {
        metadata[c].column_type = sky->column_type[c];
        metadata[c].column_attr = sky->column_attr[c];
        metadata[c].column_index = c;
    }
    qsort(
            metadata, num_columns, sizeof(oskar_SkyColumnMetadata),
            compare_column_metadata
    );

    /* Create an empty table. */
    const size_t capacity = (size_t) sky->attr_int[OSKAR_SKY_CAPACITY];
    oskar_Mem* table = oskar_mem_create(
            sky->attr_int[OSKAR_SKY_PRECISION],
            sky->attr_int[OSKAR_SKY_MEM_LOCATION],
            (size_t) num_columns * capacity, status
    );

    /* Copy the sorted columns into the new table. */
    for (c = 0; c < num_columns; ++c)
    {
        const int old_column_index = metadata[c].column_index;
        oskar_mem_copy_contents(
                table, sky->table,
                capacity * c, capacity * old_column_index, capacity, status
        );
        sky->column_type[c] = metadata[c].column_type;
        sky->column_attr[c] = metadata[c].column_attr;
    }
    free(metadata);

    /* Free the old table and keep the new one. */
    oskar_mem_free(sky->table, status);
    sky->table = table;

    /* Set all the new column references. */
    for (c = 0; c < num_columns; ++c)
    {
        oskar_mem_set_alias(
                sky->columns[c], sky->table, c * capacity, capacity, status
        );
    }
}

#ifdef __cplusplus
}
#endif
