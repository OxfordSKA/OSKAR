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

#include "synthesis/oskar_synthesis_xyz2uvw.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Single precision.

void oskar_synthesisf_xyz2uvw(int na, const float* x, const float* y,
        const float* z, double ha0, double dec0, float* u, float* v,
        float* w)
{
    double sinHa0 = sin(ha0);
    double cosHa0 = cos(ha0);
    double sinDec0 = sin(dec0);
    double cosDec0 = cos(dec0);

    int a = 0;
    for (a = 0; a < na; ++a) {
        u[a] =  x[a] * sinHa0 + y[a] * cosHa0;
        v[a] = sinDec0 * (-x[a] * cosHa0 + y[a] * sinHa0) + z[a] * cosDec0;
        w[a] = cosDec0 * (x[a] * cosHa0 - y[a] * sinHa0) + z[a] * sinDec0;
    }
}

// Double precision.

void oskar_synthesisd_xyz2uvw(int na, const double* x, const double* y,
        const double* z, double ha0, double dec0, double* u, double* v,
        double* w)
{
    double sinHa0 = sin(ha0);
    double cosHa0 = cos(ha0);
    double sinDec0 = sin(dec0);
    double cosDec0 = cos(dec0);

    int a = 0;
    for (a = 0; a < na; ++a) {
        u[a] =  x[a] * sinHa0 + y[a] * cosHa0;
        v[a] = sinDec0 * (-x[a] * cosHa0 + y[a] * sinHa0) + z[a] * cosDec0;
        w[a] = cosDec0 * (x[a] * cosHa0 - y[a] * sinHa0) + z[a] * sinDec0;
    }
}

#ifdef __cplusplus
}
#endif
