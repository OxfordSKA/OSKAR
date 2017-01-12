/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#include "convert/oskar_convert_healpix_ring_to_theta_phi.h"
#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_healpix_ring_to_theta_phi(int nside,
        oskar_Mem* theta, oskar_Mem* phi, int* status)
{
    int type, location;
    unsigned int num_pixels;

    if (*status) return;

    type = oskar_mem_type(theta);
    if (oskar_mem_type(phi) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    location = oskar_mem_location(theta);
    if (oskar_mem_location(phi) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    if (location != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    num_pixels = 12 * nside * nside;
    if (oskar_mem_length(theta) < num_pixels || oskar_mem_length(phi) < num_pixels)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    if (type == OSKAR_DOUBLE)
    {
        double *theta_, *phi_;
        unsigned int i;
        theta_ = oskar_mem_double(theta, status);
        phi_ = oskar_mem_double(phi, status);
        if (*status) return;
        for (i = 0; i < num_pixels; ++i)
        {
            oskar_convert_healpix_ring_to_theta_phi_d(nside, i, &theta_[i], &phi_[i]);
        }
    }
    else if (type == OSKAR_SINGLE)
    {
        float *theta_, *phi_;
        unsigned int i;
        theta_ = oskar_mem_float(theta, status);
        phi_ = oskar_mem_float(phi, status);
        if (*status) return;
        for (i = 0; i < num_pixels; ++i)
        {
            oskar_convert_healpix_ring_to_theta_phi_f(nside, i, &theta_[i], &phi_[i]);
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
}


void oskar_convert_healpix_ring_to_theta_phi_d(long nside, long ipix,
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

void oskar_convert_healpix_ring_to_theta_phi_f(long nside, long ipix,
        float* theta, float* phi)
{
    int npix, ncap, iring, iphi, ip;
    float fact1, fact2, fodd, hip, fihip;

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
        fihip = floorf(hip);
        /* Ring index counted from North pole. */
        iring = (int)floorf(sqrtf(hip - sqrtf(fihip))) + 1;
        iphi  = (ipix + 1) - 2 * iring * (iring - 1);

        *theta = acosf(1.0 - iring * iring / fact2);
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

        *theta = acosf((2 * nside - iring) / fact1);
        *phi   = (iphi - fodd) * M_PI / (2.0 * nside);
    }
    else
    {
        /* South polar cap. */
        ip    = npix - ipix;
        hip   = ip / 2.0;
        fihip = floorf(hip);
        /* Ring index counted from South pole. */
        iring = (int)floorf(sqrtf(hip - sqrtf(fihip))) + 1;
        iphi  = (4 * iring + 1) - (ip - 2 * iring * (iring - 1));

        *theta = acosf(-1.0 + iring * iring / fact2);
        *phi   = (iphi - 0.5) * M_PI / (2.0 * iring);
    }
}

#ifdef __cplusplus
}
#endif
