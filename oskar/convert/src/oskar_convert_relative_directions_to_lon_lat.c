/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#include "convert/oskar_convert_relative_directions_to_lon_lat.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_convert_relative_directions_to_lon_lat_2d_f(int num_points,
        const float* l, const float* m, float lon0_rad, float lat0_rad,
        float* lon_rad, float* lat_rad)
{
    int i;
    float sin_lat0, cos_lat0;
    sin_lat0 = sinf(lat0_rad);
    cos_lat0 = cosf(lat0_rad);

    /* Loop over positions and evaluate the longitude and latitude values. */
    for (i = 0; i < num_points; ++i)
    {
        float l_, m_, n_;
        l_ = l[i];
        m_ = m[i];
        n_ = sqrtf(1.0f - l_*l_ - m_*m_);
        lat_rad[i] = asinf(n_ * sin_lat0 + m_ * cos_lat0);
        lon_rad[i] = lon0_rad + atan2f(l_, cos_lat0 * n_ - m_ * sin_lat0);
    }
}

/* Double precision. */
void oskar_convert_relative_directions_to_lon_lat_2d_d(int num_points,
        const double* l, const double* m, double lon0_rad, double lat0_rad,
        double* lon_rad, double* lat_rad)
{
    int i;
    double sin_lat0, cos_lat0;
    sin_lat0 = sin(lat0_rad);
    cos_lat0 = cos(lat0_rad);

    /* Loop over positions and evaluate the longitude and latitude values. */
    for (i = 0; i < num_points; ++i)
    {
        double l_, m_, n_;
        l_ = l[i];
        m_ = m[i];
        n_ = sqrt(1.0 - l_*l_ - m_*m_);
        lat_rad[i] = asin(n_ * sin_lat0 + m_ * cos_lat0);
        lon_rad[i] = lon0_rad + atan2(l_, cos_lat0 * n_ - m_ * sin_lat0);
    }
}

#ifdef __cplusplus
}
#endif
