/*
 * Copyright (c) 2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_IMAGER_TAPER_WEIGHTS_H_
#define OSKAR_IMAGER_TAPER_WEIGHTS_H_

#include <mem/oskar_mem.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_taper_weights(size_t num_points, const oskar_Mem* uu,
        const oskar_Mem* vv, const oskar_Mem* weight_in, oskar_Mem* weight_out,
        const double uv_taper[2], int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
