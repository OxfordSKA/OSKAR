/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager.h"

#include "imager/private_imager_filter_uv.h"
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_filter_uv(oskar_Imager* h, size_t* num_vis,
        oskar_Mem* uu, oskar_Mem* vv, oskar_Mem* ww, oskar_Mem* amp,
        oskar_Mem* weight, int* status)
{
    size_t i = 0;
    double r = 0.0, range[2];

    /* Return immediately if filtering is not enabled. */
    if (h->uv_filter_min <= 0.0 &&
            (h->uv_filter_max < 0.0 || h->uv_filter_max > FLT_MAX))
    {
        return;
    }
    if (*status) return;

    /* Get the range (squared, to avoid lots of square roots later). */
    range[0] = h->uv_filter_min;
    range[1] = (h->uv_filter_max < 0.0) ? (double) FLT_MAX : h->uv_filter_max;
    range[0] *= range[0];
    range[1] *= range[1];

    /* Get the number of input points, and set the number selected to zero. */
    const size_t n = *num_vis;
    *num_vis = 0;

    /* Apply the UV baseline length filter. */
    oskar_timer_resume(h->tmr_filter);
    if (h->imager_prec == OSKAR_DOUBLE)
    {
        double2* amp_ = 0;
        double *uu_ = 0, *vv_ = 0, *ww_ = 0, *weight_ = 0;
        uu_ = oskar_mem_double(uu, status);
        vv_ = oskar_mem_double(vv, status);
        ww_ = oskar_mem_double(ww, status);
        weight_ = oskar_mem_double(weight, status);
        if (!h->coords_only)
        {
            amp_ = oskar_mem_double2(amp, status);
        }

        for (i = 0; i < n; ++i)
        {
            r = uu_[i] * uu_[i] + vv_[i] * vv_[i];
            if (r >= range[0] && r <= range[1])
            {
                uu_[*num_vis] = uu_[i];
                vv_[*num_vis] = vv_[i];
                ww_[*num_vis] = ww_[i];
                weight_[*num_vis] = weight_[i];
                if (amp_) amp_[*num_vis] = amp_[i];
                (*num_vis)++;
            }
        }
    }
    else
    {
        float2* amp_ = 0;
        float *uu_ = 0, *vv_ = 0, *ww_ = 0, *weight_ = 0;
        uu_ = oskar_mem_float(uu, status);
        vv_ = oskar_mem_float(vv, status);
        ww_ = oskar_mem_float(ww, status);
        weight_ = oskar_mem_float(weight, status);
        if (!h->coords_only)
        {
            amp_ = oskar_mem_float2(amp, status);
        }

        for (i = 0; i < n; ++i)
        {
            r = uu_[i] * uu_[i] + vv_[i] * vv_[i];
            if (r >= range[0] && r <= range[1])
            {
                uu_[*num_vis] = uu_[i];
                vv_[*num_vis] = vv_[i];
                ww_[*num_vis] = ww_[i];
                weight_[*num_vis] = weight_[i];
                if (amp_) amp_[*num_vis] = amp_[i];
                (*num_vis)++;
            }
        }
    }
    oskar_timer_pause(h->tmr_filter);
}

#ifdef __cplusplus
}
#endif
