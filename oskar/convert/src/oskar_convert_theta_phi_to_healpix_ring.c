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

#include "convert/oskar_convert_theta_phi_to_healpix_ring.h"
#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_theta_phi_to_healpix_ring(long nside, double theta,
        double phi, long *ipix)
{
    int ncap, npix, jp, jm, ipix1, ir, ip;
    double z, za, tt;

    /* Get longitude into correct range. */
    while (phi >= 2.0 * M_PI)
        phi -= 2.0 * M_PI;
    while (phi < 0.0)
        phi += 2.0 * M_PI;

    z = cos(theta);
    za = fabs(z);
    tt = phi / (0.5 * M_PI);

    /* Number of pixels in the north polar cap, and in whole map. */
    ncap  = 2 * nside * (nside - 1);
    npix  = 12 * nside * nside;

    if (za <= 2.0/3.0)
    {
        /* Equatorial region. */
        int kshift = 0;

        /* Index of ascending edge line. */
        jp = (int)floor(nside * (0.5 + tt - z * 0.75));

        /* Index of descending edge line. */
        jm = (int)floor(nside * (0.5 + tt + z * 0.75));

        /* Ring number counted from z = 2/3. */
        ir = nside + 1 + jp - jm; /* In range {1, 2n+1} */
        kshift = 1 - (ir & 1); /* kshift = 1 if ir even, 0 otherwise. */

        ip = (int)floor((jp + jm - nside + kshift + 1) / 2) + 1; /* In {1,4n} */
        if (ip > 4 * nside)
            ip -= (4 * nside);

        ipix1 = ncap + (4 * nside) * (ir - 1) + ip ;
    }
    else
    {
        /* North and south polar caps. */
        double tp, tmp;
        tp = tt - floor(tt);
        tmp = nside * sqrt(3.0 * (1.0 - za));

        /* Index of increasing edge line. */
        jp = (int)floor(tp * tmp);

        /* Index of decreasing edge line. */
        jm = (int)floor((1.0 - tp) * tmp);

        /* Ring number counted from the closest pole. */
        ir = jp + jm + 1;
        ip = (int)floor(tt * ir) + 1; /* In range {1, 4*ir} */
        if (ip > 4 * ir)
            ip -= (4 * ir);

        if (z > 0.0)
            ipix1 = 2 * ir * (ir - 1) + ip;
        else
            ipix1 = npix - 2 * ir * (ir + 1) + ip;
    }

    /* Return pixel index. */
    *ipix = ipix1 - 1;
}

#ifdef __cplusplus
}
#endif
