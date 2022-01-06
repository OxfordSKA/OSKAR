/*
 * Copyright (c) 2013-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_CONVERT_HEALPIX_RING_TO_THETA_PHI_H_
#define OSKAR_CONVERT_HEALPIX_RING_TO_THETA_PHI_H_

/**
 * @file oskar_convert_healpix_ring_to_theta_phi.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Converts HEALPix pixel IDs to angles for the HEALPix RING scheme.
 *
 * @details
 * Gives \p theta and \p phi corresponding to all pixels
 * for a parameter \p nside in the RING scheme.
 *
 * Note that \p theta is the polar angle (the colatitude) and \p phi is the
 * east longitude.
 *
 * \p nside must be in the range 1 to 8192.
 */
OSKAR_EXPORT
void oskar_convert_healpix_ring_to_theta_phi(unsigned int nside,
        oskar_Mem* theta, oskar_Mem* phi, int* status);

/**
 * @brief
 * Converts Healpix pixel ID to angles for the Healpix RING scheme.
 * (double precision)
 *
 * @details
 * Gives \p theta and \p phi corresponding to pixel \p ipix
 * for a parameter \p nside in the RING scheme.
 *
 * Note that \p theta is the polar angle (the colatitude) and \p phi is the
 * east longitude.
 *
 * \p nside must be in the range 1 to 8192, and \p ipix in the range 0 to
 * (12 * nside^2 - 1).
 */
OSKAR_EXPORT
void oskar_convert_healpix_ring_to_theta_phi_pixel(
        unsigned int nside, unsigned int ipix, double* theta, double* phi);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
