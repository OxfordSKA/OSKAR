/*
 * Copyright (c) 2023, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_GAUSSIAN_ELLIPSE_H_
#define OSKAR_GAUSSIAN_ELLIPSE_H_

/**
 * @file oskar_gaussian_ellipse.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_EXPORT
void oskar_gaussian_ellipse(
        int num_points,
        const oskar_Mem* l,
        const oskar_Mem* m,
        double x_a,
        double x_b,
        double x_c,
        double y_a,
        double y_b,
        double y_c,
        oskar_Mem* out,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
