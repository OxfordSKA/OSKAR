/*
 * Copyright (c) 2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager_taper_weights.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_taper_weights(size_t num_points, const oskar_Mem* uu,
        const oskar_Mem* vv, const oskar_Mem* weight_in, oskar_Mem* weight_out,
        const double uv_taper[2], int* status)
{
    size_t i = 0;
    if (*status) return;

    /* Size the output array. */
    oskar_mem_ensure(weight_out, num_points, status);
    if (*status) return;

    /* Scaling factors. */
    const double c_u = uv_taper[0] > 0.0 ?
            (log(0.3) / pow(uv_taper[0], 2.0)) : 0.0;
    const double c_v = uv_taper[1] > 0.0 ?
            (log(0.3) / pow(uv_taper[1], 2.0)) : 0.0;

    /* Scale weights according to (u,v) distance. */
    if (oskar_mem_precision(uu) == OSKAR_SINGLE)
    {
        const float* uu_ = oskar_mem_float_const(uu, status);
        const float* vv_ = oskar_mem_float_const(vv, status);
        const float* weight_in_ = oskar_mem_float_const(weight_in, status);
        float* weight_out_ = oskar_mem_float(weight_out, status);
        for (i = 0; i < num_points; ++i)
        {
            const double f = exp(
                    c_u * (uu_[i] * uu_[i]) +
                    c_v * (vv_[i] * vv_[i]));
            weight_out_[i] = weight_in_[i] * f;
        }
    }
    else
    {
        const double* uu_ = oskar_mem_double_const(uu, status);
        const double* vv_ = oskar_mem_double_const(vv, status);
        const double* weight_in_ = oskar_mem_double_const(weight_in, status);
        double* weight_out_ = oskar_mem_double(weight_out, status);
        for (i = 0; i < num_points; ++i)
        {
            const double f = exp(
                    c_u * (uu_[i] * uu_[i]) +
                    c_v * (vv_[i] * vv_[i]));
            weight_out_[i] = weight_in_[i] * f;
        }
    }
}

#ifdef __cplusplus
}
#endif
