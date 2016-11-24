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
 * Converts HEALPix pixel IDs to spherical angles for the HEALPix ring scheme.
 */
OSKAR_EXPORT
void oskar_convert_healpix_ring_to_theta_phi(int nside,
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
void oskar_convert_healpix_ring_to_theta_phi_d(long nside, long ipix,
        double* theta, double* phi);

/**
 * @brief
 * Converts Healpix pixel ID to angles for the Healpix RING scheme.
 * (single precision)
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
void oskar_convert_healpix_ring_to_theta_phi_f(long nside, long ipix,
        float* theta, float* phi);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_HEALPIX_RING_TO_THETA_PHI_H_ */
