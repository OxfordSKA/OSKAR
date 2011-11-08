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

#include "math/oskar_sph_to_lm.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_sph_to_lm_f(const int np, const float* lambda,
        const float* phi, const float lambda0, const float cosPhi0,
        const float sinPhi0, float* l, float* m)
{
    int i;
    for (i = 0; i < np; ++i)
    {
        float cosPhi, sinPhi, sinLambda, cosLambda, relLambda, pphi, ll, mm;
        pphi = phi[i];
        relLambda = lambda[i];
        relLambda -= lambda0;
        sinLambda = sinf(relLambda);
        cosLambda = cosf(relLambda);
        sinPhi = sinf(pphi);
        cosPhi = cosf(pphi);
        ll = cosPhi * sinLambda;
        mm = cosPhi0 * sinPhi - sinPhi0 * cosPhi * cosLambda;
        l[i] = ll;
        m[i] = mm;
    }
}

/* Double precision. */
void oskar_sph_to_lm_d(const int np, const double* lambda,
        const double* phi, const double lambda0, const double cosPhi0,
        const double sinPhi0, double* l, double* m)
{
    int i;
    for (i = 0; i < np; ++i)
    {
        double cosPhi, sinPhi, sinLambda, cosLambda, relLambda, pphi, ll, mm;
        pphi = phi[i];
        relLambda = lambda[i];
        relLambda -= lambda0;
        sinLambda = sin(relLambda);
        cosLambda = cos(relLambda);
        sinPhi = sin(pphi);
        cosPhi = cos(pphi);
        ll = cosPhi * sinLambda;
        mm = cosPhi0 * sinPhi - sinPhi0 * cosPhi * cosLambda;
        l[i] = ll;
        m[i] = mm;
    }
}

#ifdef __cplusplus
}
#endif
