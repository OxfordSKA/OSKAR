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

#ifndef OSKAR_ICRS_TO_HOR_LMN_FAST_INLINE_H_
#define OSKAR_ICRS_TO_HOR_LMN_FAST_INLINE_H_

/**
 * @file oskar_icrs_to_hor_fast_inline.h
 */

#include "oskar_global.h"
#include <math.h>

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
 * Time-dependent celestial conversion parameters.
 *
 * @details
 * This structure contains celestial data used for the coordinate
 * transformations.
 * - UT1 time as Modified Julian Date.
 * - Cosine and sine of Earth Rotation Angle.
 * - Cosine and sine of solar longitude.
 * - Coordinates of Celestial Intermediate Pole.
 *
 * All of these parameters are time-dependent.
 */
struct CelestialData
{
    double ERA[2];    ///< Cosine and sine of Earth Rotation Angle.
    double sol[2];    ///< Cosine and sine of solar longitude.
    double pole[3];   ///< Coordinates of Celestial Intermediate Pole.
    double UT1;       ///< UT1 time as Modified Julian Date.
};
typedef struct CelestialData CelestialData;

/**
 * @brief
 * Pre-computes celestial parameter data structure.
 *
 * @details
 * Populates the celestial parameters data structure, given the site data
 * and the current UT1.
 *
 * @param[in,out] c   Structure containing celestial parameter data.
 * @param[in]     lon Site longitude in radians.
 * @param[in]     UT1 The UT1 as a Modified Julian Date.
 */
__device__ __host__
inline void oskar_skyd_set_celestial_parameters_inline(CelestialData* c,
        const double lon, double UT1);

/**
 * @brief
 * Fast ICRS (loosely, J2000) equatorial to observed horizontal direction
 * cosines (double precision).
 *
 * @details
 * This function performs the transformation from ICRS (approximately J2000
 * Right Ascension and Declination) to local horizon direction cosines
 * using the site data and celestial data parameters.
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
 * @param[out] v        Three-element vector containing direction cosines.
 */
__device__ __host__
inline void oskar_icrs_to_hor_lmn_fast_inline_d(const CelestialData* c,
        const double cosLat, const double sinLat, const double pressure,
        const double ra, const double dec, double v[3]);

/**
 * @details
 * Adds the aberration vector to the given position, given the solar longitude.
 * The aberration vector is approximated as:
 *
 * \f{equation}{
 *    \textbf{V}_{A} \approx \left[\begin{array}{c}
 *                                  +0.99 \times 10^{-4} \sin \lambda \\
 *                                  -0.91 \times 10^{-4} \cos \lambda \\
 *                                  -0.40 \times 10^{-4} \cos \lambda \\
 *                            \end{array}\right]
 * \f}
 *
 * Reference: "Concise telescope pointing algorithm using IAU 2000 precepts",
 * by Patrick T. Wallace. (Bibcode 2008SPIE.7019E...7W, DOI 10.1117/12.788712)
 *
 * @param[in]     lambda  Vector containing cosine and sine of solar longitude.
 * @param[in,out] v       The (updated) 3-vector.
 */
__device__ __host__
inline void aberration_vector(const double lambda[2], double v[3])
{
    v[0] += (+0.99e-4 * lambda[1]); // sin(lambda)
    v[1] += (-0.91e-4 * lambda[0]); // cos(lambda)
    v[2] += (-0.40e-4 * lambda[0]); // cos(lambda)
}

/**
 * @details
 * Computes the coordinates of the Celestial Intermediate Pole (CIP), taking
 * into account first-order terms in precession and nutation.
 *
 * The CIP coordinates are computed as:
 *
 * \f{equation}{
 *    \textbf{V}_{CIP} \approx \left[\begin{array}{c}
 *                                  X_P + X_N \\
 *                                  Y_P + Y_N \\
 *                                  1 - X^2 / 2 \\
 *                             \end{array}\right]
 * \f}
 *
 * The precessional part of the motion is:
 *
 * \f{eqnarray}{
 *    X_P & \approx & +2.66 \times 10^{-7} t , \\
 *    Y_P & \approx & -8.14 \times 10^{-14} t^2 . \\
 * \f}
 *
 * Only the main 18.6-year nutation term is considered. Its argument is the
 * longitude of the Moon's ascending node; in radians:
 *
 * \f{equation}{
 *    \Omega \approx 2.182 - 9.242 \times 10^{-4} t.
 * \f}
 *
 * The nutation portion of the motion is:
 *
 * \f{eqnarray}{
 *    X_N & \approx & -33.2 \times 10^{-6} \sin\Omega , \\
 *    Y_N & \approx & 44.6 \times 10^{-6} \cos\Omega . \\
 * \f}
 *
 * Reference: "Concise telescope pointing algorithm using IAU 2000 precepts",
 * by Patrick T. Wallace. (Bibcode 2008SPIE.7019E...7W, DOI 10.1117/12.788712)
 *
 * @param[in] t    The (fractional) number of days elapsed since J2000.0.
 * @param[out] v   Coordinate vector of the Celestial Intermediate Pole.
 */
__device__ __host__
inline void celestial_intermediate_pole_coords(const double t, double v[3])
{
    const double omega = 2.182 - 9.242e-4 * t;
    v[0] = 2.66e-7 * t + (-33.2e-6 * sin(omega));
    v[1] = -8.14e-14 * t * t + 44.6e-6 * cos(omega);
    v[2] = 1.0 - v[0] * v[0] / 2.0;
}

/**
 * @details
 * Given UT1 as a Modified Julian Date, computes and returns the number of days
 * since J2000.0 (given by \f$ UT1 - 51544.5 \f$).
 *
 * @param[in] ut The current UT1 as a Modified Julian Date (JD - 2400000.5).
 *
 * @return The number of (fractional) days since J2000.0.
 */
__device__ __host__
inline double days_since_J2000(const double ut)
{
    return ut - 51544.5;
}

/**
 * @details
 * Computes the current local Earth Rotation Angle (ERA) at the given
 * \p longitude.
 *
 * For longitude \f$\lambda\f$ (in radians), this is:
 *
 * \f{equation}{
 *     \theta = 4.8949612 + 6.300387486755 t + \lambda
 * \f}
 *
 * Reference: "Concise telescope pointing algorithm using IAU 2000 precepts",
 * by Patrick T. Wallace. (Bibcode 2008SPIE.7019E...7W, DOI 10.1117/12.788712)
 *
 * @param[in] t         The (fractional) number of days elapsed since J2000.0.
 * @param[in] longitude The observatory longitude in radians (east-positive).
 * @param[out] theta    Vector containing cosine and sine of local ERA.
 *
 * @return The local Earth Rotation Angle, in radians.
 */
__device__ __host__
inline double earth_rotation_angle(const double t, const double longitude,
        double theta[2])
{
    const double angle = 4.8949612 + 6.300387486755 * t + longitude;
#ifdef __CUDACC__
    sincos(angle, &theta[1], &theta[0]);
#else
    theta[0] = cos(angle);
    theta[1] = sin(angle);
#endif
    return angle;
}

/**
 * @details
 * Applies the Earth-rotation correction, going from the Celestial
 * Intermediate Reference System (CIRS) to the Terrestrial
 * Intermediate Reference System (TIRS) using the Earth Rotation Angle (ERA),
 * \f$ \theta \f$.
 *
 * The transformation is given by the following rotation about the z-axis:
 *
 * \f{equation}{
 *    \textbf{V}_{h, \delta} \approx \left[\begin{array}{c}
 *                                     x \cos \theta + y \sin \theta \\
 *                                     -x \sin \theta + y \cos \theta \\
 *                                     z \\
 *                                   \end{array}\right]
 * \f}
 *
 * The z-component of the vector is unchanged by the transformation.
 *
 * @param[in] theta  Vector containing cosine and sine of local ERA.
 * @param[in,out] v  The (input CIRS/output TIRS) components.
 */
__device__ __host__
inline void earth_rotation_from_cirs(const double theta[2], double v[3])
{
    const double xa = v[0], ya = v[1];
    v[0] = xa * theta[0] + ya * theta[1];
    v[1] = -xa * theta[1] + ya * theta[0];
}

/**
 * @details
 * Apply the equatorial-to-horizon transformation using the site latitude.
 *
 * The transformation is given by the following rotation about the y-axis:
 *
 * \f{equation}{
 *    \textbf{V}_{TOPO} \approx \left[\begin{array}{c}
 *                                     x \sin \phi - z \cos \phi \\
 *                                     y \\
 *                                     x \cos \phi + z \sin \phi \\
 *                                   \end{array}\right]
 * \f}
 *
 * The y-component of the vector is unchanged by the transformation.
 *
 * @param[in] cosLat  Cosine of site latitude.
 * @param[in] sinLat  Sine of site latitude.
 * @param[in,out] v   The (input equatorial/output horizontal) components.
 */
__device__ __host__
inline void equatorial_to_horizon(const double cosLat, const double sinLat,
        double v[3])
{
    const double xa = v[0], za = v[2];
    v[0] = xa * sinLat - za * cosLat;
    v[2] = xa * cosLat + za * sinLat;
}

/**
 * @details
 * Applies the precession-nutation correction, going from the Geocentric
 * Celestial Reference System (GCRS, J2000) to Celestial Intermediate
 * Reference System (CIRS) of date.
 *
 * The transformation is
 * \f$ \textbf{v}_{CIRS} = \textbf{R}_{NPB} \cdot \textbf{v}_{GCRS} \f$,
 * and the rotation matrix can be approximated by:
 *
 * \f{equation}{
 *    \textbf{R}_{NPB} \approx \left[\begin{array}{ccc}
 *                                  1 - X^2 / 2 & 0 & -X \\
 *                                  0 & 1 & -Y \\
 *                                  X & Y & Z \\
 *                             \end{array}\right]
 * \f}
 *
 * Reference: "Concise telescope pointing algorithm using IAU 2000 precepts",
 * by Patrick T. Wallace. (Bibcode 2008SPIE.7019E...7W, DOI 10.1117/12.788712)
 *
 * @param[in] pole  The coordinates of the Celestial Intermediate Pole.
 * @param[in,out] v The (input GCRS/output CIRS) components.
 */
__device__ __host__
inline void precession_nutation_from_gcrs(const double pole[3], double v[3])
{
    const double x = v[0], y = v[1], z = v[2];
    const double X = pole[0], Y = pole[1], Z = pole[2];
    v[0] = Z * x - X * z;
    v[1] = y - Y * z;
    v[2] = Z * z + X * x + Y * y;
}

/**
 * @details
 * Computes the solar longitude for the given time \p t:
 *
 * \f{equation}{
 *     \lambda \approx 4.895 + 1.72021 \times 10^{-2} t
 * \f}
 *
 * Reference: "Concise telescope pointing algorithm using IAU 2000 precepts",
 * by Patrick T. Wallace. (Bibcode 2008SPIE.7019E...7W, DOI 10.1117/12.788712)
 *
 * @param[in] t       The (fractional) number of days elapsed since J2000.0.
 * @param[out] lambda Vector containing cosine and sine of solar longitude.
 *
 * @return The solar longitude, in radians.
 */
__device__ __host__
inline double solar_longitude(const double t, double lambda[2])
{
    const double longitude = 4.895 + 1.72021e-2 * t;
#ifdef __CUDACC__
    sincos(longitude, &lambda[1], &lambda[0]);
#else
    lambda[0] = cos(longitude);
    lambda[1] = sin(longitude);
#endif
    return longitude;
}

/**
 * @details
 * Normalises the given vector to length 1.0.
 *
 * @param[in,out] v The input and output vector components.
 */
__device__ __host__
inline void normalise(double v[3])
{
    const double s = (1.0 / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]));
    v[0] *= s;
    v[1] *= s;
    v[2] *= s;
}

__device__ __host__
inline void oskar_skyd_set_celestial_parameters_inline(CelestialData* c,
        const double lon, double UT1)
{
    // Timing.
    UT1 = days_since_J2000(UT1);
    earth_rotation_angle(UT1, lon, c->ERA);
    solar_longitude(UT1, c->sol);

    // Coordinates of Celestial Intermediate Pole (CIP).
    celestial_intermediate_pole_coords(UT1, c->pole);
}

__device__ __host__
inline void oskar_icrs_to_hor_lmn_fast_inline_d(const CelestialData* c,
        const double cosLat, const double sinLat, const double pressure,
        const double ra, const double dec, double v[3])
{
    // Convert to BCRS vector.
    double t;
#ifdef __CUDACC__
    sincos(dec, &v[2], &t);
    sincos(ra, &v[1], &v[0]);
    v[0] *= t;
    v[1] *= t;
#else
    t = cos(dec);
    v[2] = sin(dec);
    v[0] = cos(ra) * t;
    v[1] = sin(ra) * t;
#endif

    // Aberration (BCRS to GCRS).
    aberration_vector(c->sol, v);

    // Precession and nutation (GCRS to CIRS).
    precession_nutation_from_gcrs(c->pole, v);

    // Earth rotation (CIRS to topocentric equatorial).
    earth_rotation_from_cirs(c->ERA, v);

    // Equatorial to horizon.
    equatorial_to_horizon(cosLat, sinLat, v);

    // Refraction.
    if (pressure > 0.0)
        v[2] += (2.77e-7 * pressure / v[2]);

    // Renormalise.
    normalise(v);

    // Convert to direction cosines in documented coordinate system.
    // (x is East, y is North.)
    t = v[1];
    v[1] = -v[0];
    v[0] = t;
}

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_ICRS_TO_HOR_LMN_FAST_INLINE_H_ */
