/*
 * Copyright (c) 2013-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_healpix_ring_to_theta_phi.h"
#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_healpix_ring_to_theta_phi(unsigned int nside,
        oskar_Mem* theta, oskar_Mem* phi, int* status)
{
    if (*status) return;
    unsigned int i = 0;
    const unsigned int num_pixels = 12 * nside * nside;
    const int type = oskar_mem_type(theta);
    const int location = oskar_mem_location(theta);
    if (oskar_mem_type(phi) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
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
    if (oskar_mem_length(theta) < num_pixels ||
            oskar_mem_length(phi) < num_pixels)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    if (type == OSKAR_DOUBLE)
    {
        double *theta_ = 0, *phi_ = 0;
        theta_ = oskar_mem_double(theta, status);
        phi_ = oskar_mem_double(phi, status);
        if (*status) return;
        for (i = 0; i < num_pixels; ++i)
        {
            oskar_convert_healpix_ring_to_theta_phi_pixel(
                    nside, i, &theta_[i], &phi_[i]);
        }
    }
    else if (type == OSKAR_SINGLE)
    {
        float *theta_ = 0, *phi_ = 0;
        theta_ = oskar_mem_float(theta, status);
        phi_ = oskar_mem_float(phi, status);
        if (*status) return;
        for (i = 0; i < num_pixels; ++i)
        {
            double theta_rad = 0.0, phi_rad = 0.0;
            oskar_convert_healpix_ring_to_theta_phi_pixel(
                    nside, i, &theta_rad, &phi_rad);
            theta_[i] = (float) theta_rad;
            phi_[i] = (float) phi_rad;
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
}


void oskar_convert_healpix_ring_to_theta_phi_pixel(
        unsigned int nside, unsigned int ipix, double* theta, double* phi)
{
    int iring = 0, iphi = 0, ip = 0;
    double fodd = 0.0, hip = 0.0, fihip = 0.0;
    const unsigned int npix = 12 * nside * nside;
    const unsigned int ncap = 2 * nside * (nside - 1);
    const double fact1 = 1.5 * (double)nside;
    const double fact2 = 3.0 * (double)(nside * nside);

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
        fodd  = ((iring + nside) & 1) ? 1.0 : 0.5;

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
