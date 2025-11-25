/*
 * Copyright (c) 2012-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"

#ifdef __cplusplus
extern "C" {
#endif


void oskar_sky_copy_contents(
        oskar_Sky* dst,
        const oskar_Sky* src,
        int offset_dst,
        int offset_src,
        int num_sources,
        int* status
)
{
    int i = 0;
    if (*status) return;
    const int num_columns = oskar_sky_int(src, OSKAR_SKY_NUM_COLUMNS);
    for (i = 0; i < num_columns; ++i)
    {
        const oskar_SkyColumn column_type = oskar_sky_column_type(src, i);
        const int column_attribute = oskar_sky_column_attribute(src, i);
        oskar_mem_copy_contents(
                oskar_sky_column(dst, column_type, column_attribute, status),
                oskar_sky_column_const(src, column_type, column_attribute),
                offset_dst, offset_src, num_sources, status
        );
    }
}

#ifdef __cplusplus
}
#endif
