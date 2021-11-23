/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/oskar_imager_linear_to_stokes.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_linear_to_stokes(const oskar_Mem* in, oskar_Mem** out,
        int* status)
{
    size_t i = 0;
    if (*status) return;
    const size_t num = oskar_mem_length(in);

    /* Create output array or resize if required. */
    if (!*out)
    {
        *out = oskar_mem_create(oskar_mem_type(in), OSKAR_CPU, num, status);
    }
    else
    {
        oskar_mem_ensure(*out, num, status);
    }

    /* Copy or convert if required. */
    if (!oskar_mem_is_matrix(in))
    {
        /* Already Stokes I. */
        oskar_mem_copy_contents(*out, in, 0, 0, num, status);
    }
    else
    {
        if (oskar_mem_precision(in) == OSKAR_DOUBLE)
        {
            const double4c* d_ = oskar_mem_double4c_const(in, status);
            double4c* s_ = oskar_mem_double4c(*out, status);
            for (i = 0; i < num; ++i)
            {
                /* I = 0.5 (XX + YY) */
                s_[i].a.x =  0.5 * (d_[i].a.x + d_[i].d.x);
                s_[i].a.y =  0.5 * (d_[i].a.y + d_[i].d.y);
                /* Q = 0.5 (XX - YY) */
                s_[i].b.x =  0.5 * (d_[i].a.x - d_[i].d.x);
                s_[i].b.y =  0.5 * (d_[i].a.y - d_[i].d.y);
                /* U = 0.5 (XY + YX) */
                s_[i].c.x =  0.5 * (d_[i].b.x + d_[i].c.x);
                s_[i].c.y =  0.5 * (d_[i].b.y + d_[i].c.y);
                /* V = -0.5i (XY - YX) */
                s_[i].d.x =  0.5 * (d_[i].b.y - d_[i].c.y);
                s_[i].d.y = -0.5 * (d_[i].b.x - d_[i].c.x);
            }
        }
        else
        {
            const float4c* d_ = oskar_mem_float4c_const(in, status);
            float4c* s_ = oskar_mem_float4c(*out, status);
            for (i = 0; i < num; ++i)
            {
                /* I = 0.5 (XX + YY) */
                s_[i].a.x =  0.5 * (d_[i].a.x + d_[i].d.x);
                s_[i].a.y =  0.5 * (d_[i].a.y + d_[i].d.y);
                /* Q = 0.5 (XX - YY) */
                s_[i].b.x =  0.5 * (d_[i].a.x - d_[i].d.x);
                s_[i].b.y =  0.5 * (d_[i].a.y - d_[i].d.y);
                /* U = 0.5 (XY + YX) */
                s_[i].c.x =  0.5 * (d_[i].b.x + d_[i].c.x);
                s_[i].c.y =  0.5 * (d_[i].b.y + d_[i].c.y);
                /* V = -0.5i (XY - YX) */
                s_[i].d.x =  0.5 * (d_[i].b.y - d_[i].c.y);
                s_[i].d.y = -0.5 * (d_[i].b.x - d_[i].c.x);
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
