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

#ifndef OSKAR_CONVERT_RELATIVE_DIRECTIONS_TO_ENU_DIRECTIONS_INLINE_H_
#define OSKAR_CONVERT_RELATIVE_DIRECTIONS_TO_ENU_DIRECTIONS_INLINE_H_

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
OSKAR_INLINE
void oskar_convert_relative_directions_to_enu_directions_inline_f(
        float* x, float* y, float* z, const float l, const float m,
        const float n, const float cos_ha0, const float sin_ha0,
        const float cos_dec0, const float sin_dec0, const float cos_lat,
        const float sin_lat)
{
    float x_, y_, z_, t;

    x_ = l * cos_ha0 + m * sin_ha0 * sin_dec0 - n * sin_ha0 * cos_dec0;
    t = sin_lat * cos_ha0;
    y_ = -l * sin_lat * sin_ha0 +
            m * (cos_lat * cos_dec0 + t * sin_dec0) +
            n * (cos_lat * sin_dec0 - t * cos_dec0);
    t = cos_lat * cos_ha0;
    z_ = l * cos_lat * sin_ha0 +
            m * (sin_lat * cos_dec0 - t * sin_dec0) +
            n * (sin_lat * sin_dec0 + t * cos_dec0);
    *x = x_;
    *y = y_;
    *z = z_;
}

/* Double precision. */
OSKAR_INLINE
void oskar_convert_relative_directions_to_enu_directions_inline_d(
        double* x, double* y, double* z, const double l, const double m,
        const double n, const double cos_ha0, const double sin_ha0,
        const double cos_dec0, const double sin_dec0, const double cos_lat,
        const double sin_lat)
{
    double x_, y_, z_, t;

    x_ = l * cos_ha0 + m * sin_ha0 * sin_dec0 - n * sin_ha0 * cos_dec0;
    t = sin_lat * cos_ha0;
    y_ = -l * sin_lat * sin_ha0 +
            m * (cos_lat * cos_dec0 + t * sin_dec0) +
            n * (cos_lat * sin_dec0 - t * cos_dec0);
    t = cos_lat * cos_ha0;
    z_ = l * cos_lat * sin_ha0 +
            m * (sin_lat * cos_dec0 - t * sin_dec0) +
            n * (sin_lat * sin_dec0 + t * cos_dec0);
    *x = x_;
    *y = y_;
    *z = z_;
}

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_RELATIVE_DIRECTIONS_TO_ENU_DIRECTIONS_INLINE_H_ */
