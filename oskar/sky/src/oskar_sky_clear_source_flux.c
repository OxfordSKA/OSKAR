/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/private_sky.h"
#include "sky/oskar_sky.h"

#ifdef __cplusplus
extern "C" {
#endif


void oskar_sky_clear_source_flux(oskar_Sky* sky, int index, int* status)
{
    int i = 0;
    if (*status || index >= oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES)) return;

    /* Scan the columns to update those containing flux values. */
    const int num_columns = oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS);
    for (; i < num_columns; ++i)
    {
        const oskar_SkyColumn tp = sky->column_type[i];
        if ((tp >= OSKAR_SKY_SCRATCH_I_JY && tp <= OSKAR_SKY_SCRATCH_V_JY) ||
                (tp >= OSKAR_SKY_I_JY && tp <= OSKAR_SKY_V_JY))
        {
            /* Zero the value at the specified index. */
            oskar_mem_set_element_real(sky->columns[i], index, 0.0, status);
        }
    }
}

#ifdef __cplusplus
}
#endif
