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
    size_t i = 0;
    oskar_Mem* table = 0;
    if (*status || sky->attr_int[OSKAR_SKY_NUM_SOURCES] == num_sources) return;

    /* Allocate a new table of the correct size. */
    const size_t new_capacity = (size_t) num_sources + 1;
    const size_t old_capacity = (size_t) (sky->attr_int[OSKAR_SKY_CAPACITY]);
    const size_t num_columns = (size_t) (sky->attr_int[OSKAR_SKY_NUM_COLUMNS]);
    table = oskar_mem_create(
            sky->attr_int[OSKAR_SKY_PRECISION],
            sky->attr_int[OSKAR_SKY_MEM_LOCATION],
            new_capacity * num_columns, status
    );
    oskar_mem_clear_contents(table, status);

    /* Copy the existing data to the new table. */
    const size_t copy_len = (
            old_capacity < new_capacity ? old_capacity : new_capacity
    );
    for (i = 0; i < num_columns; ++i)
    {
        /* Copy each column. */
        oskar_mem_copy_contents(
                table, sky->table, i * new_capacity, i * old_capacity,
                copy_len, status
        );

        /* Set new column reference. */
        oskar_mem_set_alias(
                sky->columns[i], table, i * new_capacity, new_capacity, status
        );
    }

    /* Free the old table, and store the new one. */
    oskar_mem_free(sky->table, status);
    sky->table = table;

    /* Update the attributes. */
    sky->attr_int[OSKAR_SKY_CAPACITY] = (int) new_capacity;
    sky->attr_int[OSKAR_SKY_NUM_SOURCES] = num_sources;
}

#ifdef __cplusplus
}
#endif
