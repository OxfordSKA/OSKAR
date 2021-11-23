/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_append(oskar_Sky* dst, const oskar_Sky* src, int* status)
{
    int num_dst = 0, num_src = 0;
    if (*status) return;

    /* Resize the sky model. */
    num_dst = oskar_sky_num_sources(dst);
    num_src = oskar_sky_num_sources(src);
    oskar_sky_resize(dst, num_dst + num_src, status);

    /* Copy memory contents at the appropriate offset. */
    oskar_sky_copy_contents(dst, src, num_dst, 0, num_src, status);

    /* Set flag to use extended sources. */
    oskar_sky_set_use_extended(dst,
            oskar_sky_use_extended(src) || oskar_sky_use_extended(dst));
}

#ifdef __cplusplus
}
#endif
