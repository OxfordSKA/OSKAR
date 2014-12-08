/*
 * Copyright (c) 2014, The University of Oxford
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

#ifndef OSKAR_CONVERT_ECEF_TO_STATION_UVW_INLINE_H_
#define OSKAR_CONVERT_ECEF_TO_STATION_UVW_INLINE_H_

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
OSKAR_INLINE
void oskar_convert_ecef_to_station_uvw_inline_f(const float x, const float y,
        const float z, const float sin_ha0, const float cos_ha0,
        const float sin_dec0, const float cos_dec0, float* u, float* v,
        float* w)
{
    float v_, w_, t;

    /* This is just the standard textbook rotation matrix. */
    t = x * cos_ha0;
    t -= y * sin_ha0;
    v_ = z * cos_dec0;
    v_ -= sin_dec0 * t;
    w_ = cos_dec0 * t;
    w_ += z * sin_dec0;
    t =  x * sin_ha0;
    t += y * cos_ha0;
    *u = t;
    *v = v_;
    *w = w_;
}

/* Double precision. */
OSKAR_INLINE
void oskar_convert_ecef_to_station_uvw_inline_d(const double x, const double y,
        const double z, const double sin_ha0, const double cos_ha0,
        const double sin_dec0, const double cos_dec0, double* u, double* v,
        double* w)
{
    double v_, w_, t;

    /* This is just the standard textbook rotation matrix. */
    t = x * cos_ha0;
    t -= y * sin_ha0;
    v_ = z * cos_dec0;
    v_ -= sin_dec0 * t;
    w_ = cos_dec0 * t;
    w_ += z * sin_dec0;
    t =  x * sin_ha0;
    t += y * cos_ha0;
    *u = t;
    *v = v_;
    *w = w_;
}

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_ECEF_TO_STATION_UVW_INLINE_H_ */
