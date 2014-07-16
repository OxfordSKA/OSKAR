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

#ifndef OSKAR_CONVERT_RELATIVE_DIRECTION_COSINES_TO_APPARENT_RA_DEC_H_
#define OSKAR_CONVERT_RELATIVE_DIRECTION_COSINES_TO_APPARENT_RA_DEC_H_

/**
 * @file oskar_convert_relative_direction_cosines_to_apparent_ra_dec.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Convert direction cosines to angles (single precision).
 *
 * @details
 * Returns the Right Ascension (longitude) and Declination (latitude) of the
 * supplied array of direction cosines (l and m).
 *
 * Note: This is a a general direction_cosines to lon_lat conversion.
 *
 * @param[in]  np   Number of positions to evaluate.
 * @param[in]  ra0  Right Ascension of the field centre, in radians.
 * @param[in]  dec0 Declination of the field centre, in radians.
 * @param[in]  l    Array of x-positions in cosine space.
 * @param[in]  m    Array of y-positions in cosine space.
 * @param[out] ra   Array of Right Ascensions values, in radians.
 * @param[out] dec  Array of Declinations values, in radians.
 */
OSKAR_EXPORT
void oskar_convert_relative_direction_cosines_to_apparent_ra_dec_f(int np,
        float ra0, float dec0, const float* l, const float* m, float* ra,
        float* dec);

/**
 * @brief
 * Convert direction cosines to angles (double precision).
 *
 * @details
 * Returns the Right Ascension (longitude) and Declination (latitude) of the
 * supplied array of direction cosines (l and m).
 *
 * Note: This is a a general direction_cosines to lon_lat conversion.
 *
 * @param[in]  np   Number of positions to evaluate.
 * @param[in]  ra0  Right Ascension of the field centre, in radians.
 * @param[in]  dec0 Declination of the field centre, in radians.
 * @param[in]  l    Array of x-positions in cosine space.
 * @param[in]  m    Array of y-positions in cosine space.
 * @param[out] ra   Array of Right Ascensions values, in radians.
 * @param[out] dec  Array of Declinations values, in radians.
 */
OSKAR_EXPORT
void oskar_convert_relative_direction_cosines_to_apparent_ra_dec_d(int np,
        double ra0, double dec0, const double* l, const double* m, double* ra,
        double* dec);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_RELATIVE_DIRECTION_COSINES_TO_APPARENT_RA_DEC_H_ */
