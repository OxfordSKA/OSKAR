/*
 * Copyright (c) 2015, The University of Oxford
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

#ifndef OSKAR_PRIVATE_RANDOM_HELPERS_H_
#define OSKAR_PRIVATE_RANDOM_HELPERS_H_

#include <oskar_global.h>
#include <math/oskar_cmath.h>
#include <math/private_random_generators.h>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_INLINE
float oskar_int_to_range_0_to_1_f(const unsigned long in)
{
    const float factor = 1.0f / (1.0f + 0xFFFFFFFFuL);
    const float half_factor = 0.5f * factor;
    return (in * factor) + half_factor;
}

OSKAR_INLINE
double oskar_int_to_range_0_to_1_d(const unsigned long in)
{
    const double factor = 1.0 / (1.0 + 0xFFFFFFFFuL);
    const double half_factor = 0.5 * factor;
    return (in * factor) + half_factor;
}

OSKAR_INLINE
float oskar_int_to_range_minus_1_to_1_f(const unsigned long in)
{
    const float factor = 1.0f / (1.0f + 0x7FFFFFFFuL);
    const float half_factor = 0.5f * factor;
    return (((long)in) * factor) + half_factor;
}

OSKAR_INLINE
double oskar_int_to_range_minus_1_to_1_d(const unsigned long in)
{
    const double factor = 1.0 / (1.0 + 0x7FFFFFFFuL);
    const double half_factor = 0.5 * factor;
    return (((long)in) * factor) + half_factor;
}

OSKAR_INLINE
void oskar_box_muller_f(unsigned long u0, unsigned long u1,
        float* f0, float* f1)
{
    float r;
#ifdef __CUDACC__
    sincospif(oskar_int_to_range_minus_1_to_1_f(u0), f0, f1);
#else
    float t = (float) M_PI;
    t *= oskar_int_to_range_minus_1_to_1_f(u0);
    *f0 = sinf(t);
    *f1 = cosf(t);
#endif
    r = sqrtf(-2.0f * logf(oskar_int_to_range_0_to_1_f(u1)));
    *f0 *= r;
    *f1 *= r;
}

OSKAR_INLINE
void oskar_box_muller_d(unsigned long u0, unsigned long u1,
        double* f0, double* f1)
{
    double r;
#ifdef __CUDACC__
    sincospi(oskar_int_to_range_minus_1_to_1_d(u0), f0, f1);
#else
    double t = M_PI;
    t *= oskar_int_to_range_minus_1_to_1_d(u0);
    *f0 = sin(t);
    *f1 = cos(t);
#endif
    r = sqrt(-2.0 * log(oskar_int_to_range_0_to_1_d(u1)));
    *f0 *= r;
    *f1 *= r;
}

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_PRIVATE_RANDOM_HELPERS_H_ */
