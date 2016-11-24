/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include "sky/private_sky.h"
#include "sky/oskar_sky.h"

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_rotate_to_position(oskar_Sky* sky,
        double ra0, double dec0, int* status)
{
    /*
     * The rotation matrix is given by:
     *
     *   [ cos(a)sin(d)   -sin(a)   cos(a)cos(d) ]
     *   [ sin(a)sin(d)    cos(a)   sin(a)cos(d) ]
     *   [    -cos(d)        0          sin(d)   ]
     *
     * where a = ra0, d = dec0.
     * This corresponds to a rotation of a around z,
     * followed by a rotation of (90-d) around y.
     */

    int i, type, location;

    /* Check if safe to proceed. */
    if (*status) return;

    type = oskar_sky_precision(sky);
    location = oskar_sky_mem_location(sky);
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            /* Construct rotation matrix. */
            float *ra, *dec;
            float cosA, sinA, cosD, sinD;
            float m11, m12, m13, m21, m22, m23, m31, m33;
            cosA = cosf(ra0);
            sinA = sinf(ra0);
            cosD = cosf(dec0);
            sinD = sinf(dec0);
            m11 = cosA * sinD; m12 = -sinA; m13 = cosA * cosD;
            m21 = sinA * sinD; m22 =  cosA; m23 = sinA * cosD;
            m31 = -cosD; m33 = sinD;
            ra = oskar_mem_float(oskar_sky_ra_rad(sky), status);
            dec = oskar_mem_float(oskar_sky_dec_rad(sky), status);

            /* Loop over current sources. */
            for (i = 0; i < sky->num_sources; ++i)
            {
                float x, y, z, a, b, c;

                /* Get direction cosines. */
                c = cosf(dec[i]);
                x = c * cosf(ra[i]);
                y = c * sinf(ra[i]);
                z = sinf(dec[i]);

                /* Apply rotation matrix. */
                a = m11 * x + m12 * y + m13 * z;
                b = m21 * x + m22 * y + m23 * z;
                c = m31 * x + m33 * z;

                /* Convert back to angles. */
                ra[i] = atan2f(b, a);
                dec[i] = atan2f(c, sqrtf(a*a + b*b));
            }
        }
        else if (type == OSKAR_DOUBLE)
        {
            /* Construct rotation matrix. */
            double *ra, *dec;
            double cosA, sinA, cosD, sinD;
            double m11, m12, m13, m21, m22, m23, m31, m33;
            cosA = cos(ra0);
            sinA = sin(ra0);
            cosD = cos(dec0);
            sinD = sin(dec0);
            m11 = cosA * sinD; m12 = -sinA; m13 = cosA * cosD;
            m21 = sinA * sinD; m22 =  cosA; m23 = sinA * cosD;
            m31 = -cosD; m33 = sinD;
            ra = oskar_mem_double(oskar_sky_ra_rad(sky), status);
            dec = oskar_mem_double(oskar_sky_dec_rad(sky), status);

            /* Loop over current sources. */
            for (i = 0; i < sky->num_sources; ++i)
            {
                double x, y, z, a, b, c;

                /* Get direction cosines. */
                c = cos(dec[i]);
                x = c * cos(ra[i]);
                y = c * sin(ra[i]);
                z = sin(dec[i]);

                /* Apply rotation matrix. */
                a = m11 * x + m12 * y + m13 * z;
                b = m21 * x + m22 * y + m23 * z;
                c = m31 * x + m33 * z;

                /* Convert back to angles. */
                ra[i] = atan2(b, a);
                dec[i] = atan2(c, sqrt(a*a + b*b));
            }
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else
        *status = OSKAR_ERR_BAD_LOCATION;
}

#ifdef __cplusplus
}
#endif
