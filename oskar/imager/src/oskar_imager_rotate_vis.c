/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_cmath.h"
#include "imager/private_imager.h"
#include "imager/oskar_imager.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_rotate_vis(oskar_Imager* h, size_t num_vis,
        const oskar_Mem* uu_in, const oskar_Mem* vv_in, const oskar_Mem* ww_in,
        oskar_Mem* amps)
{
#ifdef OSKAR_OS_WIN
    int i;
    const int num = (const int) num_vis;
#else
    size_t i = 0;
    const size_t num = num_vis;
#endif
    const double delta_l = h->delta_l;
    const double delta_m = h->delta_m;
    const double delta_n = h->delta_n;
    const double twopi = 2.0 * M_PI;

    oskar_timer_resume(h->tmr_rotate);
    if (oskar_mem_precision(amps) == OSKAR_DOUBLE)
    {
        const double *u = 0, *v = 0, *w = 0;
        double2* a = 0;
        u = (const double*)oskar_mem_void_const(uu_in);
        v = (const double*)oskar_mem_void_const(vv_in);
        w = (const double*)oskar_mem_void_const(ww_in);
        a = (double2*)oskar_mem_void(amps);

#pragma omp parallel for private(i)
        for (i = 0; i < num; ++i)
        {
            double arg = 0.0, phase_re = 0.0, phase_im = 0.0, re = 0.0, im = 0.0;
            arg = twopi * (u[i] * delta_l + v[i] * delta_m + w[i] * delta_n);
            phase_re = cos(arg);
            phase_im = sin(arg);
            re = a[i].x * phase_re - a[i].y * phase_im;
            im = a[i].x * phase_im + a[i].y * phase_re;
            a[i].x = re;
            a[i].y = im;
        }
    }
    else
    {
        const float *u = 0, *v = 0, *w = 0;
        float2* a = 0;
        u = (const float*)oskar_mem_void_const(uu_in);
        v = (const float*)oskar_mem_void_const(vv_in);
        w = (const float*)oskar_mem_void_const(ww_in);
        a = (float2*)oskar_mem_void(amps);

#pragma omp parallel for private(i)
        for (i = 0; i < num; ++i)
        {
            double arg = 0.0, phase_re = 0.0, phase_im = 0.0, re = 0.0, im = 0.0;
            arg = twopi * (u[i] * delta_l + v[i] * delta_m + w[i] * delta_n);
            phase_re = cos(arg);
            phase_im = sin(arg);
            re = a[i].x * phase_re - a[i].y * phase_im;
            im = a[i].x * phase_im + a[i].y * phase_re;
            a[i].x = (float) re;
            a[i].y = (float) im;
        }
    }
    oskar_timer_pause(h->tmr_rotate);
}

#ifdef __cplusplus
}
#endif
