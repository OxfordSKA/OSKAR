/*
 * Copyright (c) 2011-2014, The University of Oxford
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


#include "math/oskar_sph_rotate_to_position.h"
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_sph_rotate_to_position(int n, oskar_Mem* lon, oskar_Mem* lat,
        double lon0, double lat0)
{
    int i;

    if (lon == NULL || lat == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (oskar_mem_location(lon) != OSKAR_CPU || oskar_mem_location(lat) != OSKAR_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    if (oskar_mem_type(lon) == OSKAR_DOUBLE && oskar_mem_type(lat) == OSKAR_DOUBLE)
    {
        /* Construct rotation matrix. */
        double cosLon, sinLon, cosLat, sinLat;
        double m11, m12, m13, m21, m22, m23, m31, m33;
        cosLon = cos(lon0);
        sinLon = sin(lon0);
        cosLat = cos(lat0);
        sinLat = sin(lat0);
        m11 = cosLon * sinLat; m12 = -sinLon; m13 = cosLon * cosLat;
        m21 = sinLon * sinLat; m22 =  cosLon; m23 = sinLon * cosLat;
        m31 = -cosLat; m33 = sinLat;

        for (i = 0; i < n; ++i)
        {
            double x, y, z, a, b, c;

            /* Direction cosines */
            c = cos(((double*)lat->data)[i]);
            x = c * cos(((double*)lon->data)[i]);
            y = c * sin(((double*)lon->data)[i]);
            z = sin(((double*)lat->data)[i]);

            /* Apply rotation matrix */
            a = m11 * x + m12 * y + m13 * z;
            b = m21 * x + m22 * y + m23 * z;
            c = m31 * x + m33 * z;

            /* Convert back to angles. */
            ((double*)lon->data)[i] = atan2(b, a);
            ((double*)lat->data)[i] = atan2(c, sqrt(a*a + b*b));
        }
    }
    else if (oskar_mem_type(lon) == OSKAR_SINGLE && oskar_mem_type(lat) == OSKAR_SINGLE)
    {
        /* Construct rotation matrix. */
        float cosLon, sinLon, cosLat, sinLat;
        float m11, m12, m13, m21, m22, m23, m31, m33;
        cosLon = cosf(lon0);
        sinLon = sinf(lon0);
        cosLat = cosf(lat0);
        sinLat = sinf(lat0);
        m11 = cosLon * sinLat; m12 = -sinLon; m13 = cosLon * cosLat;
        m21 = sinLon * sinLat; m22 =  cosLon; m23 = sinLon * cosLat;
        m31 = -cosLat; m33 = sinLat;

        for (i = 0; i < n; ++i)
        {
            float x, y, z, a, b, c;

            /* Direction cosines */
            c = cosf(((float*)lat->data)[i]);
            x = c * cosf(((float*)lon->data)[i]);
            y = c * sinf(((float*)lon->data)[i]);
            z = sinf(((float*)lat->data)[i]);

            /* Apply rotation matrix */
            a = m11 * x + m12 * y + m13 * z;
            b = m21 * x + m22 * y + m23 * z;
            c = m31 * x + m33 * z;

            /* Convert back to angles. */
            ((float*)lon->data)[i] = atan2f(b, a);
            ((float*)lat->data)[i] = atan2f(c, sqrtf(a*a + b*b));
        }
    }
    else
    {
        return OSKAR_ERR_BAD_DATA_TYPE;
    }

    return 0;
}


#ifdef __cplusplus
}
#endif
