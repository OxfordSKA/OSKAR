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

#include <oskar_convert_apparent_ha_dec_to_enu_directions.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_convert_apparent_ha_dec_to_enu_directions_f(int n,
        const float* ha, const float* dec, float lat, float* x, float* y,
        float* z)
{
    float sinLat, cosLat;
    int i;

    /* Compute latitude trig. */
    sinLat = sinf(lat);
    cosLat = cosf(lat);

    /* Loop over positions. */
    for (i = 0; i < n; ++i)
    {
        float sh, sd, sinHA, sinDec, cosHA, cosDec, t, X1, Y2;

        /* Load local equatorial coordinates. */
        sh = ha[i];
        sd = dec[i];

        /* Find direction cosines. */
        sinHA = sinf(sh);
        sinDec = sinf(sd);
        cosHA = cosf(sh);
        cosDec = cosf(sd);
        t = cosDec * cosHA;
        X1 = cosLat * sinDec - sinLat * t;
        Y2 = sinLat * sinDec + cosLat * t;
        t = -cosDec * sinHA;

        /* Store source direction cosines. */
        x[i] = t;  /* Horizontal x-component. */
        y[i] = X1; /* Horizontal y-component. */
        z[i] = Y2; /* Horizontal z-component. */
    }
}

/* Double precision. */
void oskar_convert_apparent_ha_dec_to_enu_directions_d(int n,
        const double* ha, const double* dec, double lat, double* x, double* y,
        double* z)
{
    double sinLat, cosLat;
    int i;

    /* Compute latitude trig. */
    sinLat = sin(lat);
    cosLat = cos(lat);

    /* Loop over positions. */
    for (i = 0; i < n; ++i)
    {
        double sh, sd, sinHA, sinDec, cosHA, cosDec, t, X1, Y2;

        /* Load local equatorial coordinates. */
        sh = ha[i];
        sd = dec[i];

        /* Find direction cosines. */
        sinHA = sin(sh);
        sinDec = sin(sd);
        cosHA = cos(sh);
        cosDec = cos(sd);
        t = cosDec * cosHA;
        X1 = cosLat * sinDec - sinLat * t;
        Y2 = sinLat * sinDec + cosLat * t;
        t = -cosDec * sinHA;

        /* Store source direction cosines. */
        x[i] = t;  /* Horizontal x-component. */
        y[i] = X1; /* Horizontal y-component. */
        z[i] = Y2; /* Horizontal z-component. */
    }
}

#ifdef __cplusplus
}
#endif
