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

#include "oskar_convert_healpix_ring_to_theta_phi.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_healpix_ring_to_theta_phi(long nside, long ipix,
        double* theta, double* phi)
{
    int npix, ncap, iring, iphi, ip;
    double fact1, fact2, fodd, hip, fihip;

    /* Compute total number of points. */
    npix = 12 * nside * nside;

    /* Factors. */
    fact1 = 1.5 * nside;
    fact2 = 3.0 * nside * nside;

    /* Points in each polar cap. */
    ncap = 2 * nside * (nside - 1);

    /* Check which region the pixel is in. */
    if (ipix < ncap)
    {
        /* North polar cap. */
        hip   = (ipix + 1) / 2.0;
        fihip = floor(hip);
        /* Ring index counted from North pole. */
        iring = (int)floor( sqrt(hip - sqrt(fihip)) ) + 1;
        iphi  = (ipix + 1) - 2 * iring * (iring - 1);

        *theta = acos(1.0 - iring * iring / fact2);
        *phi   = (iphi - 0.5) * M_PI / (2.0 * iring);
    }
    else if (ipix < (npix - ncap))
    {
        /* Equatorial region. */
        ip    = ipix - ncap;
        /* Ring index counted from North pole. */
        iring = ip / (4 * nside) + nside;
        iphi  = ip % (4 * nside) + 1;
        /* fodd = 1 if iring + nside is odd, 0.5 otherwise */
        fodd  = ((iring + nside) & 1) ? 1 : 0.5;

        *theta = acos((2 * nside - iring) / fact1);
        *phi   = (iphi - fodd) * M_PI / (2.0 * nside);
    }
    else
    {
        /* South polar cap. */
        ip    = npix - ipix;
        hip   = ip / 2.0;
        fihip = floor(hip);
        /* Ring index counted from South pole. */
        iring = (int)floor( sqrt(hip - sqrt(fihip)) ) + 1;
        iphi  = (4 * iring + 1) - (ip - 2 * iring * (iring - 1));

        *theta = acos(-1.0 + iring * iring / fact2);
        *phi   = (iphi - 0.5) * M_PI / (2.0 * iring);
    }
}


#ifdef __cplusplus
}
#endif
