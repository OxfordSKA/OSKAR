/*
 * Copyright (c) 2013-2017, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "sky/oskar_update_horizon_mask.h"
#include "sky/oskar_update_horizon_mask_cuda.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_update_horizon_mask(int num_sources, const oskar_Mem* l,
        const oskar_Mem* m, const oskar_Mem* n, const double ha0_rad,
        const double dec0_rad, const double lat_rad, oskar_Mem* mask,
        int* status)
{
    int i, type, location, *mask_;
    double cos_ha0, sin_dec0, cos_dec0, sin_lat, cos_lat;
    double ll, mm, nn;
    if (*status) return;
    type = oskar_mem_precision(l);
    location = oskar_mem_location(mask);
    mask_ = oskar_mem_int(mask, status);
    cos_ha0  = cos(ha0_rad);
    sin_dec0 = sin(dec0_rad);
    cos_dec0 = cos(dec0_rad);
    sin_lat  = sin(lat_rad);
    cos_lat  = cos(lat_rad);
    ll = cos_lat * sin(ha0_rad);
    mm = sin_lat * cos_dec0 - cos_lat * cos_ha0 * sin_dec0;
    nn = sin_lat * sin_dec0 + cos_lat * cos_ha0 * cos_dec0;
    switch (type)
    {
    case OSKAR_SINGLE:
    {
        float ll_, mm_, nn_;
        const float *l_, *m_, *n_;
        l_ = oskar_mem_float_const(l, status);
        m_ = oskar_mem_float_const(m, status);
        n_ = oskar_mem_float_const(n, status);
        ll_ = (float) ll;
        mm_ = (float) mm;
        nn_ = (float) nn;
        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_update_horizon_mask_cuda_f(num_sources, l_, m_, n_,
                    ll_, mm_, nn_, mask_);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_CPU)
        {
            for (i = 0; i < num_sources; ++i)
                mask_[i] |= ((l_[i] * ll_ + m_[i] * mm_ + n_[i] * nn_) > 0.);
        }
        else
            *status = OSKAR_ERR_BAD_LOCATION;
        break;
    }
    case OSKAR_DOUBLE:
    {
        const double *l_, *m_, *n_;
        l_ = oskar_mem_double_const(l, status);
        m_ = oskar_mem_double_const(m, status);
        n_ = oskar_mem_double_const(n, status);
        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_update_horizon_mask_cuda_d(num_sources, l_, m_, n_,
                    ll, mm, nn, mask_);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_CPU)
        {
            for (i = 0; i < num_sources; ++i)
                mask_[i] |= ((l_[i] * ll + m_[i] * mm + n_[i] * nn) > 0.);
        }
        else
            *status = OSKAR_ERR_BAD_LOCATION;
        break;
    }
    default:
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        break;
    }
}

#ifdef __cplusplus
}
#endif
