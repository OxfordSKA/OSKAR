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

#include <math.h>
#include <private_correlate_functions_inline.h>
#include <oskar_auto_correlate_omp.h>
#include <oskar_add_inline.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_auto_correlate_omp_f(const int num_sources, const int num_stations,
        const float4c* jones, const float* source_I, const float* source_Q,
        const float* source_U, const float* source_V, float4c* vis)
{
    int s;

    /* Loop over stations. */
#pragma omp parallel for private(s)
    for (s = 0; s < num_stations; ++s)
    {
        int i;
        const float4c *station;
        float4c sum, guard;

        oskar_clear_complex_matrix_f(&sum);
        oskar_clear_complex_matrix_f(&guard);

        /* Pointer to source vector for station. */
        station = &jones[s * num_sources];

        /* Accumulate visibility response for source. */
        for (i = 0; i < num_sources; ++i)
            oskar_accumulate_station_visibility_for_source_inline_f(&sum,
                    i, source_I, source_Q, source_U, source_V, station,
                    &guard);

        /* Add result to the station visibility. */
        /* Blank non-Hermitian values. */
        sum.a.y = 0.0f;
        sum.d.y = 0.0f;
        oskar_add_complex_matrix_in_place_f(&vis[s], &sum);
    }
}

/* Double precision. */
void oskar_auto_correlate_omp_d(const int num_sources, const int num_stations,
        const double4c* jones, const double* source_I, const double* source_Q,
        const double* source_U, const double* source_V, double4c* vis)
{
    int s;

    /* Loop over stations. */
#pragma omp parallel for private(s)
    for (s = 0; s < num_stations; ++s)
    {
        int i;
        const double4c *station;
        double4c sum;

        oskar_clear_complex_matrix_d(&sum);

        /* Pointer to source vector for station. */
        station = &jones[s * num_sources];

        /* Accumulate visibility response for source. */
        for (i = 0; i < num_sources; ++i)
            oskar_accumulate_station_visibility_for_source_inline_d(&sum,
                    i, source_I, source_Q, source_U, source_V, station);

        /* Add result to the station visibility. */
        /* Blank non-Hermitian values. */
        sum.a.y = 0.0;
        sum.d.y = 0.0;
        oskar_add_complex_matrix_in_place_d(&vis[s], &sum);
    }
}

#ifdef __cplusplus
}
#endif
