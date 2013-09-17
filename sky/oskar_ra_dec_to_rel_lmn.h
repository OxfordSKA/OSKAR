/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#ifndef OSKAR_RA_DEC_TO_REL_LMN_H_
#define OSKAR_RA_DEC_TO_REL_LMN_H_

/**
 * @file oskar_ra_dec_to_rel_lmn.h
 */

#include <oskar_global.h>
#include <oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Equatorial to relative 3D direction cosines (single precision).
 *
 * @details
 * This function computes the l,m,n direction cosines of the specified points
 * relative to the reference point.
 *
 * Note that the n-positions are given by sqrt(1 - l*l - m*m) - 1.
 *
 * @param[in] num_points The number of points.
 * @param[in] h_ra       The input position Right Ascensions in radians.
 * @param[in] h_dec      The input position Declinations in radians.
 * @param[in] ra0_rad    The Right Ascension of the reference point in radians.
 * @param[in] dec0_rad   The Declination of the reference point in radians.
 * @param[out] h_l       The l-direction-cosines relative to the reference point.
 * @param[out] h_m       The m-direction-cosines relative to the reference point.
 * @param[out] h_n       The n-direction-cosines relative to the reference point.
 */
OSKAR_EXPORT
void oskar_ra_dec_to_rel_lmn_f(int num_points, const float* h_ra,
        const float* h_dec, float ra0_rad, float dec0_rad, float* h_l,
        float* h_m, float* h_n);

/**
 * @brief
 * Equatorial to relative 3D direction cosines (double precision).
 *
 * @details
 * This function computes the l,m,n direction cosines of the specified points
 * relative to the reference point.
 *
 * Note that the n-positions are given by sqrt(1 - l*l - m*m) - 1.
 *
 * @param[in] num_points The number of points.
 * @param[in] h_ra       The input position Right Ascensions in radians.
 * @param[in] h_dec      The input position Declinations in radians.
 * @param[in] ra0_rad    The Right Ascension of the reference point in radians.
 * @param[in] dec0_rad   The Declination of the reference point in radians.
 * @param[out] h_l       The l-direction-cosines relative to the reference point.
 * @param[out] h_m       The m-direction-cosines relative to the reference point.
 * @param[out] h_n       The n-direction-cosines relative to the reference point.
 */
OSKAR_EXPORT
void oskar_ra_dec_to_rel_lmn_d(int num_points, const double* h_ra,
        const double* h_dec, double ra0_rad, double dec0_rad, double* h_l,
        double* h_m, double* h_n);

/**
 * @brief
 * Equatorial to relative 3D direction cosines (wrapper function).
 *
 * @details
 * This function computes the l,m,n direction cosines of the specified points
 * relative to the reference point.
 *
 * Note that the n-positions are given by sqrt(1 - l*l - m*m) - 1.
 *
 * @param[in] num_points The number of points.
 * @param[in] ra         The input position Right Ascensions in radians.
 * @param[in] dec        The input position Declinations in radians.
 * @param[in] ra0_rad    The Right Ascension of the reference point in radians.
 * @param[in] dec0_rad   The Declination of the reference point in radians.
 * @param[out] l         The l-direction-cosines relative to the reference point.
 * @param[out] m         The m-direction-cosines relative to the reference point.
 * @param[out] n         The n-direction-cosines relative to the reference point.
 */
OSKAR_EXPORT
void oskar_ra_dec_to_rel_lmn(int num_points, const oskar_Mem* ra,
        const oskar_Mem* dec, double ra0_rad, double dec0_rad, oskar_Mem* l,
        oskar_Mem* m, oskar_Mem* n,  int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_RA_DEC_TO_REL_LMN_H_ */
