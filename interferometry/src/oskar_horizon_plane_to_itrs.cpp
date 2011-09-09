/*
 * Copyright (c) 2011, The University of Oxford
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

#include "interferometry/oskar_horizon_plane_to_itrs.h"

#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_horizon_plane_to_itrs_d(const unsigned num_antennas,
        const double * x_horizon, const double * y_horizon,
        const double latitude, double * x, double * y, double * z)
{
    const double sinLat = sin(latitude);
    const double cosLat = cos(latitude);

    for (unsigned i = 0; i < num_antennas; ++i)
    {
        x[i] = -y_horizon[i] * sinLat;
        y[i] =  x_horizon[i];
        z[i] =  y_horizon[i] * cosLat;
    }
}


void oskar_horizon_plane_to_itrs_f(const unsigned num_antennas,
        const float* x_horizon, const float* y_horizon,
        const float latitude, float* x, float* y, float* z)
{
    const float sinLat = sinf(latitude);
    const float cosLat = cosf(latitude);

    for (unsigned i = 0; i < num_antennas; ++i)
    {
        x[i] = -y_horizon[i] * sinLat;
        y[i] =  x_horizon[i];
        z[i] =  y_horizon[i] * cosLat;
    }
}

#ifdef __cplusplus
}
#endif
