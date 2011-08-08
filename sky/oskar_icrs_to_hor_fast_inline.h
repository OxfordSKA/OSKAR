/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_ICRS_TO_HOR_FAST_INLINE_H_
#define OSKAR_ICRS_TO_HOR_FAST_INLINE_H_

/**
 * @file oskar_icrs_to_hor_fast_inline.h
 */

#include "sky/oskar_icrs_to_hor_lmn_fast_inline.h"
#include "utility/oskar_util_cuda_eclipse.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

/**
 * @brief
 * Fast ICRS (loosely, J2000) equatorial to observed horizontal coordinates
 * (double precision).
 *
 * @details
 * This function performs the transformation from ICRS (approximately J2000
 * Right Ascension and Declination) to local horizon coordinates (azimuth and
 * elevation) using the site data and celestial data parameters.
 *
 * It allows for:
 *
 * - Annual relativistic aberration.
 * - Precession.
 * - Nutation.
 * - Earth rotation.
 * - Site location.
 * - Atmospheric refraction.
 *
 * The effects neglected are:
 *
 * - Light deflection (under 2 arcsec even at Solar limb).
 * - Frame bias (negligible).
 * - Polar motion (this is below 0.5 arcsec).
 * - Diurnal aberration and parallax (this is below 0.3 arcsec).
 *
 * The approximations in these routines produce results that are accurate to
 * less than 2 arcsec for the whole of the 21st Century. The RMS error is less
 * than 1 arcsec.
 *
 * Reference: "Concise telescope pointing algorithm using IAU 2000 precepts",
 * by Patrick T. Wallace. (Bibcode 2008SPIE.7019E...7W, DOI 10.1117/12.788712)
 *
 * @param[in]  c        Populated structure containing time-dependent data.
 * @param[in]  cosLat   Cosine of site latitude.
 * @param[in]  sinLat   Sine of site latitude.
 * @param[in]  pressure Air pressure in millibars.
 * @param[in]  ra       The ICRS (J2000) Right Ascension in radians.
 * @param[in]  dec      The ICRS (J2000) Declination in radians.
 * @param[out] az       The local azimuth in radians.
 * @param[out] el       The local elevation in radians.
 */
__host__ __device__
inline void oskar_icrs_to_hor_fast_inline_d(const CelestialData* c,
        const double cosLat, const double sinLat, const double pressure,
        const double ra, const double dec, double* az, double* el);


__host__ __device__
inline void oskar_icrs_to_hor_fast_inline_d(const CelestialData* c,
        const double cosLat, const double sinLat, const double pressure,
        const double ra, const double dec, double* az, double* el)
{
    // Call the routine that converts to direction cosines.
    double v[3];
    oskar_icrs_to_hor_lmn_fast_inline_d(c, cosLat, sinLat, pressure,
            ra, dec, v);

    // Convert to angles.
    *az = atan2(v[0], v[1]);
    *el = atan2(v[2], sqrt(v[0] * v[0] + v[1] * v[1]));
}

#ifdef __cplusplus
}
#endif

#endif // OSKAR_ICRS_TO_HOR_FAST_INLINE_H_
