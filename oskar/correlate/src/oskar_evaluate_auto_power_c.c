/*
 * Copyright (c) 2015-2018, The University of Oxford
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

#include "correlate/oskar_evaluate_auto_power_c.h"
#include "correlate/private_correlate_functions_inline.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_evaluate_auto_power_f(const int num_sources,
        const float4c* restrict jones, float4c* restrict out)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        float4c val1, val2;
        OSKAR_LOAD_MATRIX(val1, jones[i]);
        val2 = val1; /* Auto-power product. */
        OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(float2, val1, val2)
        out[i] = val1;
    }
}

void oskar_evaluate_auto_power_scalar_f(const int num_sources,
        const float2* restrict jones, float2* restrict out)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        float2 val = jones[i];
        val.x = val.x * val.x + val.y * val.y; /* Auto-power product. */
        val.y = 0.0f;
        out[i] = val;
    }
}

/* Double precision. */
void oskar_evaluate_auto_power_d(const int num_sources,
        const double4c* restrict jones, double4c* restrict out)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        double4c val1, val2;
        OSKAR_LOAD_MATRIX(val1, jones[i]);
        val2 = val1; /* Auto-power product. */
        OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(double2, val1, val2)
        out[i] = val1;
    }
}

void oskar_evaluate_auto_power_scalar_d(const int num_sources,
        const double2* restrict jones, double2* restrict out)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        double2 val = jones[i];
        val.x = val.x * val.x + val.y * val.y; /* Auto-power product. */
        val.y = 0.0;
        out[i] = val;
    }
}

#ifdef __cplusplus
}
#endif
