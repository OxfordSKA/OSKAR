/*
 * Copyright (c) 2013, The University of Oxford
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

#include <oskar_convert_apparent_ha_dec_to_az_el.h>
#include <oskar_convert_apparent_ha_dec_to_enu_direction_cosines.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_convert_apparent_ha_dec_to_az_el_f(int n, const float* ha,
        const float* dec, float lat, float* work, float* az, float* el)
{
    int i;
    oskar_convert_apparent_ha_dec_to_enu_direction_cosines_f(n, ha, dec, lat,
            work, az, el);
    for (i = 0; i < n; ++i)
    {
        float x, y, z, a;
        x = work[i];
        y = az[i];
        z = el[i];
        a = atan2f(x, y); /* Azimuth. */
        x = sqrtf(x*x + y*y);
        y = atan2f(z, x); /* Elevation. */
        az[i] = a;
        el[i] = y;
    }
}

/* Double precision. */
void oskar_convert_apparent_ha_dec_to_az_el_d(int n, const double* ha,
        const double* dec, double lat, double* work, double* az, double* el)
{
    int i;
    oskar_convert_apparent_ha_dec_to_enu_direction_cosines_d(n, ha, dec, lat,
            work, az, el);
    for (i = 0; i < n; ++i)
    {
        double x, y, z, a;
        x = work[i];
        y = az[i];
        z = el[i];
        a = atan2(x, y); /* Azimuth. */
        x = sqrt(x*x + y*y);
        y = atan2(z, x); /* Elevation. */
        az[i] = a;
        el[i] = y;
    }
}


#ifdef __cplusplus
}
#endif
