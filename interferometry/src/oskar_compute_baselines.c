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

#include "interferometry/oskar_compute_baselines.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single precision.
void oskar_compute_baselines_f(int na, const float* au,
        const float* av, const float* aw, float* bu, float* bv, float* bw)
{
    int a1, a2, b; // Station and baseline indices.
    for (a1 = 0, b = 0; a1 < na; ++a1)
    {
        for (a2 = a1 + 1; a2 < na; ++a2, ++b)
        {
            bu[b] = au[a2] - au[a1];
            bv[b] = av[a2] - av[a1];
            bw[b] = aw[a2] - aw[a1];
        }
    }
}

// Double precision.
void oskar_compute_baselines_d(int na, const double* au,
        const double* av, const double* aw, double* bu, double* bv, double* bw)
{
    int a1, a2, b; // Station and baseline indices.
    for (a1 = 0, b = 0; a1 < na; ++a1)
    {
        for (a2 = a1 + 1; a2 < na; ++a2, ++b)
        {
            bu[b] = au[a2] - au[a1];
            bv[b] = av[a2] - av[a1];
            bw[b] = aw[a2] - aw[a1];
        }
    }
}

#ifdef __cplusplus
}
#endif
