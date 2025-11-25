/*
 * Copyright (c) 2015-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"
#include "sky/private_sky.h"

#ifdef __cplusplus
extern "C" {
#endif


void oskar_sky_copy(oskar_Sky* dst, const oskar_Sky* src, int* status)
{
    int i = 0;
    if (*status) return;
    if (oskar_sky_int(dst, OSKAR_SKY_PRECISION) !=
            oskar_sky_int(src, OSKAR_SKY_PRECISION))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;                /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }
    const int num_sources = oskar_sky_int(src, OSKAR_SKY_NUM_SOURCES);
    if (oskar_sky_int(dst, OSKAR_SKY_CAPACITY) < num_sources)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;           /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }
    for (i = 0; i < OSKAR_SKY_NUM_ATTRIBUTES_INT; ++i)
    {
        const oskar_SkyAttribInt attr = (oskar_SkyAttribInt) i;
        oskar_sky_set_int(dst, attr, oskar_sky_int(src, attr));
    }
    for (i = 0; i < OSKAR_SKY_NUM_ATTRIBUTES_DOUBLE; ++i)
    {
        const oskar_SkyAttribDouble attr = (oskar_SkyAttribDouble) i;
        oskar_sky_set_double(dst, attr, oskar_sky_double(src, attr));
    }
    oskar_sky_copy_contents(dst, src, 0, 0, num_sources, status);
}

#ifdef __cplusplus
}
#endif
