/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager_weight_radial.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_weight_radial(size_t num_points, const oskar_Mem* uu,
        const oskar_Mem* vv, const oskar_Mem* weight_in, oskar_Mem* weight_out,
        int* status)
{
    size_t i = 0;
    if (*status) return;
    oskar_mem_realloc(weight_out, num_points, status);
    if (oskar_mem_precision(weight_out) == OSKAR_DOUBLE)
    {
        double *wt_out = 0;
        const double *u = 0, *v = 0, *wt_in = 0;
        u      = oskar_mem_double_const(uu, status);
        v      = oskar_mem_double_const(vv, status);
        wt_in  = oskar_mem_double_const(weight_in, status);
        wt_out = oskar_mem_double(weight_out, status);
        for (i = 0; i < num_points; ++i)
        {
            wt_out[i] = wt_in[i] * sqrt(u[i]*u[i] + v[i]*v[i]);
        }
    }
    else
    {
        float *wt_out = 0;
        const float *u = 0, *v = 0, *wt_in = 0;
        u      = oskar_mem_float_const(uu, status);
        v      = oskar_mem_float_const(vv, status);
        wt_in  = oskar_mem_float_const(weight_in, status);
        wt_out = oskar_mem_float(weight_out, status);
        for (i = 0; i < num_points; ++i)
        {
            wt_out[i] = wt_in[i] * sqrt(u[i]*u[i] + v[i]*v[i]);
        }
    }
}

#ifdef __cplusplus
}
#endif
