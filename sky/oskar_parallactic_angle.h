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

#ifndef OSKAR_PARALLACTIC_ANGLE_H_
#define OSKAR_PARALLACTIC_ANGLE_H_

/**
 * @file oskar_parallactic_angle.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Function to compute the parallactic angle at a position (single precision).
 *
 * @details
 * This function computes the parallactic angle at a position on the sky,
 * at a given latitude.
 *
 * @param[in] ha       The hour angle.
 * @param[in] dec      The declination.
 * @param[in] cos_lat  Cosine of the observer's latitude.
 * @param[in] sin_lat  Sine of the observer's latitude.
 */
OSKAR_INLINE
float oskar_parallactic_angle_f(const float ha, const float dec,
        const float cos_lat, const float sin_lat)
{
    float sin_dec, cos_dec, sin_ha, cos_ha, x, y;
#ifdef __CUDACC__
    sincosf(ha, &sin_ha, &cos_ha);
    sincosf(dec, &sin_dec, &cos_dec);
#else
    sin_ha = sinf(ha);
    cos_ha = cosf(ha);
    sin_dec = sinf(dec);
    cos_dec = cosf(dec);
#endif
    y = cos_lat * sin_ha;
    x = sin_lat * cos_dec - cos_lat * sin_dec * cos_ha;
    return atan2f(y, x);
}

/**
 * @brief
 * Function to compute the parallactic angle at a position (double precision).
 *
 * @details
 * This function computes the parallactic angle at a position on the sky,
 * at a given latitude.
 *
 * @param[in] ha       The hour angle.
 * @param[in] dec      The declination.
 * @param[in] cos_lat  Cosine of the observer's latitude.
 * @param[in] sin_lat  Sine of the observer's latitude.
 */
OSKAR_INLINE
double oskar_parallactic_angle_d(const double ha, const double dec,
        const double cos_lat, const double sin_lat)
{
    double sin_dec, cos_dec, sin_ha, cos_ha, x, y;
#ifdef __CUDACC__
    sincos(ha, &sin_ha, &cos_ha);
    sincos(dec, &sin_dec, &cos_dec);
#else
    sin_ha = sin(ha);
    cos_ha = cos(ha);
    sin_dec = sin(dec);
    cos_dec = cos(dec);
#endif
    y = cos_lat * sin_ha;
    x = sin_lat * cos_dec - cos_lat * sin_dec * cos_ha;
    return atan2(y, x);
}

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_PARALLACTIC_ANGLE_H_ */
