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

#ifndef OSKAR_PRIVATE_COND2_2X2_H_
#define OSKAR_PRIVATE_COND2_2X2_H_

#include <math.h>

#include "oskar_global.h"
#include "math/define_multiply.h"
#include "utility/oskar_vector_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Private inline functions. */

OSKAR_INLINE
float oskar_cond2_2x2_inline_f(const float4c* RESTRICT in)
{
    float sum, diff, t1, t2, a, b;
    float4c p, q;
    p = *in;
    q = p;
    OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(float2, p, q)
    sum = p.a.x + p.d.x;
    t1  = p.a.x - p.d.x;
    t1  = t1 * t1;
    t2  = 4.0f * ((p.b.x * p.b.x) + (p.b.y * p.b.y));
    diff = sqrtf(t1 + t2);
    a = sqrtf(0.5f * (sum + diff));
    b = sqrtf(0.5f * (sum - diff));
    return (a > b) ? a / b : b / a;
}

OSKAR_INLINE
double oskar_cond2_2x2_inline_d(const double4c* RESTRICT in)
{
    double sum, diff, t1, t2, a, b;
    double4c p, q;
    p = *in;
    q = p;
    OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(double2, p, q)
    sum = p.a.x + p.d.x;
    t1  = p.a.x - p.d.x;
    t1  = t1 * t1;
    t2  = 4.0 * ((p.b.x * p.b.x) + (p.b.y * p.b.y));
    diff = sqrt(t1 + t2);
    a = sqrt(0.5 * (sum + diff));
    b = sqrt(0.5 * (sum - diff));
    return (a > b) ? a / b : b / a;
}

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_PRIVATE_COND2_2X2_H_ */
