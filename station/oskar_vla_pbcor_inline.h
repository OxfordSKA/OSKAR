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

#ifndef OSKAR_VLA_PBCOR_INLINE_H_
#define OSKAR_VLA_PBCOR_INLINE_H_

/**
 * @file oskar_vla_pbcor_inline.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates a scalar VLA dish beam, as implemented in the AIPS task PBCOR
 * (single precision).
 *
 * @details
 * This function evaluates a scalar VLA dish beam, as implemented in the AIPS
 * task PBCOR.
 *
 * See http://www.aips.nrao.edu/cgi-bin/ZXHLP2.PL?PBCOR
 *
 * @param[in]  l             Direction cosine of each source from phase centre.
 * @param[in]  m             Direction cosine of each source from phase centre.
 * @param[in]  freq_ghz      Current observing frequency in GHz.
 * @param[in]  p1            Value of PBPARM(3) for this frequency.
 * @param[in]  p2            Value of PBPARM(4) for this frequency.
 * @param[in]  p3            Value of PBPARM(5) for this frequency.
 *
 * @return Value of VLA beam.
 */
OSKAR_INLINE
float oskar_vla_pbcor_inline_f(const float l, const float m,
        const float freq_ghz, const float p1, const float p2,
        const float p3)
{
    float r, t, X, cutoff_arcmin;
    if (l != l) return l; /* Catch and return NAN without using the macro. */
    cutoff_arcmin = 44.376293f / freq_ghz;
    r = asinf(sqrtf(l * l + m * m)) * 3437.74677078493951f; /* rad to arcmin */
    if (r < cutoff_arcmin)
    {
        t = r * freq_ghz;
        X = t * t;
        return 1.0f + X * (p1 * 1e-3f + X * (p2 * 1e-7f + X * p3 * 1e-10f));
    }
    return 0.0f;
}

/**
 * @brief
 * Evaluates a scalar VLA dish beam, as implemented in the AIPS task PBCOR
 * (double precision).
 *
 * @details
 * This function evaluates a scalar VLA dish beam, as implemented in the AIPS
 * task PBCOR.
 *
 * See http://www.aips.nrao.edu/cgi-bin/ZXHLP2.PL?PBCOR
 *
 * @param[in]  l             Direction cosine of each source from phase centre.
 * @param[in]  m             Direction cosine of each source from phase centre.
 * @param[in]  freq_ghz      Current observing frequency in GHz.
 * @param[in]  p1            Value of PBPARM(3) for this frequency.
 * @param[in]  p2            Value of PBPARM(4) for this frequency.
 * @param[in]  p3            Value of PBPARM(5) for this frequency.
 *
 * @return Value of VLA beam.
 */
OSKAR_INLINE
double oskar_vla_pbcor_inline_d(const double l, const double m,
        const double freq_ghz, const double p1, const double p2,
        const double p3)
{
    double r, t, X, cutoff_arcmin;
    if (l != l) return l; /* Catch and return NAN without using the macro. */
    cutoff_arcmin = 44.376293 / freq_ghz;
    r = asin(sqrt(l * l + m * m)) * 3437.74677078493951; /* rad to arcmin */
    if (r < cutoff_arcmin)
    {
        t = r * freq_ghz;
        X = t * t;
        return 1.0 + X * (p1 * 1e-3 + X * (p2 * 1e-7 + X * p3 * 1e-10));
    }
    return 0.0;
}

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_VLA_PBCOR_INLINE_H_ */
