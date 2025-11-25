/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"

#ifdef __cplusplus
extern "C" {
#endif


void oskar_sky_append(oskar_Sky* dst, const oskar_Sky* src, int* status)
{
    if (*status) return;

    /* Resize the sky model. */
    const int num_dst = oskar_sky_int(dst, OSKAR_SKY_NUM_SOURCES);
    const int num_src = oskar_sky_int(src, OSKAR_SKY_NUM_SOURCES);
    oskar_sky_resize(dst, num_dst + num_src, status);

    /* Copy memory contents at the appropriate offset. */
    oskar_sky_copy_contents(dst, src, num_dst, 0, num_src, status);

    /* Set flag to use extended sources. */
    oskar_sky_set_int(
            dst, OSKAR_SKY_USE_EXTENDED,
            oskar_sky_int(src, OSKAR_SKY_USE_EXTENDED) ||
            oskar_sky_int(dst, OSKAR_SKY_USE_EXTENDED)
    );
}

#ifdef __cplusplus
}
#endif
