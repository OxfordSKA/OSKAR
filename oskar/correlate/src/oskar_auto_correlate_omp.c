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

#include "correlate/private_correlate_functions_inline.h"
#include "correlate/oskar_auto_correlate_omp.h"
#include "math/oskar_add_inline.h"
#include "math/oskar_kahan_sum.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_auto_correlate_omp_f(const int num_sources, const int num_stations,
        const float4c* jones, const float* source_I, const float* source_Q,
        const float* source_U, const float* source_V, float4c* vis)
{
    int s;
#pragma omp parallel for private(s)
    for (s = 0; s < num_stations; ++s)
    {
        int i;
        float4c m1, m2, sum, guard;
        const float4c *const jones_station = &jones[s * num_sources];
        OSKAR_CLEAR_COMPLEX_MATRIX(float, sum)
        OSKAR_CLEAR_COMPLEX_MATRIX(float, guard)
        for (i = 0; i < num_sources; ++i)
        {
            /* Construct source brightness matrix. */
            OSKAR_CONSTRUCT_B(float, m2,
                    source_I[i], source_Q[i], source_U[i], source_V[i])

            /* Multiply first Jones matrix with source brightness matrix. */
            OSKAR_LOAD_MATRIX(m1, jones_station[i])
            OSKAR_MUL_COMPLEX_MATRIX_HERMITIAN_IN_PLACE(float2, m1, m2);

            /* Multiply result with second (Hermitian transposed) Jones matrix. */
            OSKAR_LOAD_MATRIX(m2, jones_station[i])
            OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(float2, m1, m2);

            /* Accumulate. */
            OSKAR_KAHAN_SUM_COMPLEX_MATRIX(float, sum, m1, guard)
        }

        /* Blank non-Hermitian values. */
        sum.a.y = 0.0f; sum.d.y = 0.0f;
        OSKAR_ADD_COMPLEX_MATRIX_IN_PLACE(vis[s], sum);
    }
}

/* Double precision. */
void oskar_auto_correlate_omp_d(const int num_sources, const int num_stations,
        const double4c* jones, const double* source_I, const double* source_Q,
        const double* source_U, const double* source_V, double4c* vis)
{
    int s;
#pragma omp parallel for private(s)
    for (s = 0; s < num_stations; ++s)
    {
        int i;
        double4c m1, m2, sum;
        const double4c *const jones_station = &jones[s * num_sources];
        OSKAR_CLEAR_COMPLEX_MATRIX(double, sum)
        for (i = 0; i < num_sources; ++i)
        {
            /* Construct source brightness matrix. */
            OSKAR_CONSTRUCT_B(double, m2,
                    source_I[i], source_Q[i], source_U[i], source_V[i])

            /* Multiply first Jones matrix with source brightness matrix. */
            OSKAR_LOAD_MATRIX(m1, jones_station[i])
            OSKAR_MUL_COMPLEX_MATRIX_HERMITIAN_IN_PLACE(double2, m1, m2);

            /* Multiply result with second (Hermitian transposed) Jones matrix. */
            OSKAR_LOAD_MATRIX(m2, jones_station[i])
            OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(double2, m1, m2);

            /* Accumulate. */
            OSKAR_ADD_COMPLEX_MATRIX_IN_PLACE(sum, m1)
        }

        /* Blank non-Hermitian values. */
        sum.a.y = 0.0; sum.d.y = 0.0;
        OSKAR_ADD_COMPLEX_MATRIX_IN_PLACE(vis[s], sum);
    }
}

#ifdef __cplusplus
}
#endif
