/*
 * Copyright (c) 2013-2019, The University of Oxford
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
#include "utility/oskar_device.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_update_horizon_mask(int num_sources, const oskar_Mem* l,
        const oskar_Mem* m, const oskar_Mem* n, const double ha0_rad,
        const double dec0_rad, const double lat_rad, oskar_Mem* mask,
        int* status)
{
    if (*status) return;
    const int type = oskar_mem_precision(l);
    const int location = oskar_mem_location(mask);
    const double cos_ha0  = cos(ha0_rad);
    const double sin_dec0 = sin(dec0_rad);
    const double cos_dec0 = cos(dec0_rad);
    const double sin_lat  = sin(lat_rad);
    const double cos_lat  = cos(lat_rad);
    const double ll = cos_lat * sin(ha0_rad);
    const double mm = sin_lat * cos_dec0 - cos_lat * cos_ha0 * sin_dec0;
    const double nn = sin_lat * sin_dec0 + cos_lat * cos_ha0 * cos_dec0;
    const float ll_ = (float) ll, mm_ = (float) mm, nn_ = (float) nn;
    if (location == OSKAR_CPU)
    {
        int i, *mask_;
        mask_ = oskar_mem_int(mask, status);
        if (type == OSKAR_SINGLE)
        {
            const float *l_, *m_, *n_;
            l_ = oskar_mem_float_const(l, status);
            m_ = oskar_mem_float_const(m, status);
            n_ = oskar_mem_float_const(n, status);
            for (i = 0; i < num_sources; ++i)
                mask_[i] |= ((l_[i] * ll_ + m_[i] * mm_ + n_[i] * nn_) > 0.f);
        }
        else if (type == OSKAR_DOUBLE)
        {
            const double *l_, *m_, *n_;
            l_ = oskar_mem_double_const(l, status);
            m_ = oskar_mem_double_const(m, status);
            n_ = oskar_mem_double_const(n, status);
            for (i = 0; i < num_sources; ++i)
                mask_[i] |= ((l_[i] * ll + m_[i] * mm + n_[i] * nn) > 0.);
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        const int is_dbl = (type == OSKAR_DOUBLE);
        if (type == OSKAR_DOUBLE)      k = "update_horizon_mask_double";
        else if (type == OSKAR_SINGLE) k = "update_horizon_mask_float";
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_sources, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &num_sources},
                {PTR_SZ, oskar_mem_buffer_const(l)},
                {PTR_SZ, oskar_mem_buffer_const(m)},
                {PTR_SZ, oskar_mem_buffer_const(n)},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&ll : (const void*)&ll_},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&mm : (const void*)&mm_},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&nn : (const void*)&nn_},
                {PTR_SZ, oskar_mem_buffer(mask)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
