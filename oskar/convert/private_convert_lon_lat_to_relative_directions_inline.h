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

#ifndef OSKAR_PRIVATE_CONVERT_LON_LAT_TO_RELATIVE_DIRECTIONS_INLINE_H_
#define OSKAR_PRIVATE_CONVERT_LON_LAT_TO_RELATIVE_DIRECTIONS_INLINE_H_

#include <oskar_global.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
OSKAR_INLINE
void oskar_convert_lon_lat_to_relative_directions_inline_f(float lon_rad,
        const float lat_rad, const float lon0_rad, const float cos_lat0,
        const float sin_lat0, float* l, float* m, float* n)
{
    float sin_lon, cos_lon, sin_lat, cos_lat, l_, m_, n_;

    /* Convert from spherical to tangent-plane. */
    lon_rad -= lon0_rad;
#ifdef __CUDACC__
    sincosf(lon_rad, &sin_lon, &cos_lon);
    sincosf(lat_rad, &sin_lat, &cos_lat);
#else
    sin_lon = sinf(lon_rad);
    cos_lon = cosf(lon_rad);
    sin_lat = sinf(lat_rad);
    cos_lat = cosf(lat_rad);
#endif
    l_ = cos_lat * sin_lon;
    m_ = cos_lat0 * sin_lat - sin_lat0 * cos_lat * cos_lon;
    n_ = sin_lat0 * sin_lat + cos_lat0 * cos_lat * cos_lon;

    /* Store output data. */
    *l = l_;
    *m = m_;
    *n = n_;
}

/* Double precision. */
OSKAR_INLINE
void oskar_convert_lon_lat_to_relative_directions_inline_d(double lon_rad,
        const double lat_rad, const double lon0_rad, const double cos_lat0,
        const double sin_lat0, double* l, double* m, double* n)
{
    double sin_lon, cos_lon, sin_lat, cos_lat, l_, m_, n_;

    /* Convert from spherical to tangent-plane. */
    lon_rad -= lon0_rad;
#ifdef __CUDACC__
    sincos(lon_rad, &sin_lon, &cos_lon);
    sincos(lat_rad, &sin_lat, &cos_lat);
#else
    sin_lon = sin(lon_rad);
    cos_lon = cos(lon_rad);
    sin_lat = sin(lat_rad);
    cos_lat = cos(lat_rad);
#endif
    l_ = cos_lat * sin_lon;
    m_ = cos_lat0 * sin_lat - sin_lat0 * cos_lat * cos_lon;
    n_ = sin_lat0 * sin_lat + cos_lat0 * cos_lat * cos_lon;

    /* Store output data. */
    *l = l_;
    *m = m_;
    *n = n_;
}

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_PRIVATE_CONVERT_LON_LAT_TO_RELATIVE_DIRECTIONS_INLINE_H_ */
