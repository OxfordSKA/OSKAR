/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager.h"
#include "imager/oskar_imager.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_rotate_coords(oskar_Imager* h, size_t num_coords,
        const oskar_Mem* uu_in, const oskar_Mem* vv_in, const oskar_Mem* ww_in,
        oskar_Mem* uu_out, oskar_Mem* vv_out, oskar_Mem* ww_out)
{
#ifdef OSKAR_OS_WIN
    int i;
    const int num = (const int) num_coords;
#else
    size_t i = 0;
    const size_t num = num_coords;
#endif
    const double *M = h->M;
    oskar_timer_resume(h->tmr_rotate);
    if (oskar_mem_precision(uu_in) == OSKAR_SINGLE)
    {
        float *uu_o = 0, *vv_o = 0, *ww_o = 0;
        const float *uu_i = 0, *vv_i = 0, *ww_i = 0;
        uu_i = (const float*)oskar_mem_void_const(uu_in);
        vv_i = (const float*)oskar_mem_void_const(vv_in);
        ww_i = (const float*)oskar_mem_void_const(ww_in);
        uu_o = (float*)oskar_mem_void(uu_out);
        vv_o = (float*)oskar_mem_void(vv_out);
        ww_o = (float*)oskar_mem_void(ww_out);

#pragma omp parallel for private(i)
        for (i = 0; i < num; ++i)
        {
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, t0 = 0.0, t1 = 0.0, t2 = 0.0;
            s0 = uu_i[i]; s1 = vv_i[i]; s2 = ww_i[i];
            t0 = M[0] * s0 + M[1] * s1 + M[2] * s2;
            t1 = M[3] * s0 + M[4] * s1 + M[5] * s2;
            t2 = M[6] * s0 + M[7] * s1 + M[8] * s2;
            uu_o[i] = t0; vv_o[i] = t1; ww_o[i] = t2;
        }
    }
    else
    {
        double *uu_o = 0, *vv_o = 0, *ww_o = 0;
        const double *uu_i = 0, *vv_i = 0, *ww_i = 0;
        uu_i = (const double*)oskar_mem_void_const(uu_in);
        vv_i = (const double*)oskar_mem_void_const(vv_in);
        ww_i = (const double*)oskar_mem_void_const(ww_in);
        uu_o = (double*)oskar_mem_void(uu_out);
        vv_o = (double*)oskar_mem_void(vv_out);
        ww_o = (double*)oskar_mem_void(ww_out);

#pragma omp parallel for private(i)
        for (i = 0; i < num; ++i)
        {
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, t0 = 0.0, t1 = 0.0, t2 = 0.0;
            s0 = uu_i[i]; s1 = vv_i[i]; s2 = ww_i[i];
            t0 = M[0] * s0 + M[1] * s1 + M[2] * s2;
            t1 = M[3] * s0 + M[4] * s1 + M[5] * s2;
            t2 = M[6] * s0 + M[7] * s1 + M[8] * s2;
            uu_o[i] = t0; vv_o[i] = t1; ww_o[i] = t2;
        }
    }
    oskar_timer_pause(h->tmr_rotate);
}

#ifdef __cplusplus
}
#endif
