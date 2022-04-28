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
    const unsigned int num_pix = 12 * nside * nside;
    const unsigned int num_in_cap = 2 * nside * (nside - 1);

    /* Check which region the pixel is in. */
    if (ipix < num_in_cap)
    {
        /* North polar cap. Ring index from north. */
        const double i_pix_half = 0.5 * (ipix + 1);
        const int i_ring = (int)floor(
                sqrt(i_pix_half - sqrt(floor(i_pix_half)))) + 1;
        const int i_phi = (ipix + 1) - 2 * i_ring * (i_ring - 1);
        *theta = acos(1.0 - i_ring * i_ring / (3.0 * nside * nside));
        *phi = (i_phi - 0.5) * M_PI / (2.0 * i_ring);
    }
    else if (ipix < (num_pix - num_in_cap))
    {
        /* Equatorial region. Ring index from north. */
        const int i_pix_local = ipix - num_in_cap;
        const int i_ring = i_pix_local / (4 * nside) + nside;
        const int i_phi = i_pix_local % (4 * nside) + 1;
        const double f = ((i_ring + nside) & 1) ? 1.0 : 0.5;
        *theta = acos((2.0 * nside - i_ring) / (1.5 * nside));
        *phi = (i_phi - f) * M_PI / (2.0 * nside);
    }
    else
    {
        /* South polar cap. Ring index from south. */
        const int i_pix_local = num_pix - ipix;
        const double i_pix_half = 0.5 * i_pix_local;
        const int i_ring = (int)floor(
                sqrt(i_pix_half - sqrt(floor(i_pix_half)))) + 1;
        const int i_phi = (4 * i_ring + 1) -
                (i_pix_local - 2 * i_ring * (i_ring - 1));
        *theta = acos(-1.0 + i_ring * i_ring / (3.0 * nside * nside));
        *phi = (i_phi - 0.5) * M_PI / (2.0 * i_ring);
    }
}

#ifdef __cplusplus
}
#endif
