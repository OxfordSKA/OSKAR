/*
 * Copyright (c) 2014, The University of Oxford
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

#ifndef OSKAR_CONVERT_APPARENT_RA_DEC_TO_RELATIVE_DIRECTION_COSINES_INLINE_H_
#define OSKAR_CONVERT_APPARENT_RA_DEC_TO_RELATIVE_DIRECTION_COSINES_INLINE_H_

#include <oskar_global.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
OSKAR_INLINE
void oskar_convert_apparent_ra_dec_to_relative_direction_cosines_inline_f(
        float ra, const float dec, const float ra0, const float cos_dec0,
        const float sin_dec0, float* l, float* m, float* n)
{
    float cos_dec, sin_dec, sin_ra, cos_ra, l_, m_, n_;

    /* Convert from spherical to tangent-plane. */
    ra -= ra0;
#ifdef __CUDACC__
    sincosf(ra, &sin_ra, &cos_ra);
    sincosf(dec, &sin_dec, &cos_dec);
#else
    sin_ra = sinf(ra);
    cos_ra = cosf(ra);
    sin_dec = sinf(dec);
    cos_dec = cosf(dec);
#endif
    l_ = cos_dec * sin_ra;
    m_ = cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_ra;
    n_ = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_ra;

    /* Store output data. */
    *l = l_;
    *m = m_;
    *n = n_;
}

/* Double precision. */
OSKAR_INLINE
void oskar_convert_apparent_ra_dec_to_relative_direction_cosines_inline_d(
        double ra, const double dec, const double ra0, const double cos_dec0,
        const double sin_dec0, double* l, double* m, double* n)
{
    double cos_dec, sin_dec, sin_ra, cos_ra, l_, m_, n_;

    /* Convert from spherical to tangent-plane. */
    ra -= ra0;
#ifdef __CUDACC__
    sincos(ra, &sin_ra, &cos_ra);
    sincos(dec, &sin_dec, &cos_dec);
#else
    sin_ra = sin(ra);
    cos_ra = cos(ra);
    sin_dec = sin(dec);
    cos_dec = cos(dec);
#endif
    l_ = cos_dec * sin_ra;
    m_ = cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_ra;
    n_ = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_ra;

    /* Store output data. */
    *l = l_;
    *m = m_;
    *n = n_;
}

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_APPARENT_RA_DEC_TO_RELATIVE_DIRECTION_COSINES_INLINE_H_ */
