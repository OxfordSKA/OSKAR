/*
 * Copyright (c) 2014-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_ellipse_radius.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

double oskar_ellipse_radius(double maj_axis, double min_axis,
        double pa_ellipse_rad, double pa_point_rad)
{
    double l = 0.0, m = 0.0;
    const double sin_pa = sin(pa_ellipse_rad);
    const double cos_pa = cos(pa_ellipse_rad);
    const double sin_b = sin(pa_point_rad - pa_ellipse_rad);
    const double cos_b = cos(pa_point_rad - pa_ellipse_rad);
    l = 0.5 * (maj_axis * cos_b * sin_pa + min_axis * sin_b * cos_pa);
    m = 0.5 * (maj_axis * cos_b * cos_pa - min_axis * sin_b * sin_pa);
    return sqrt(l*l + m*m);
}

#ifdef __cplusplus
}
#endif
