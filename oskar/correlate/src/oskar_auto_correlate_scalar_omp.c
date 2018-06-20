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

#include "correlate/oskar_auto_correlate_scalar_omp.h"
#include "math/oskar_kahan_sum.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_auto_correlate_scalar_omp_f(const int num_sources,
        const int num_stations, const float2* jones, const float* source_I,
        float2* vis)
{
    int i, s;
#pragma omp parallel for private(i, s)
    for (s = 0; s < num_stations; ++s)
    {
        float sum = 0.0f, guard = 0.0f;
        const float2 *const jones_station = &jones[s * num_sources];
        for (i = 0; i < num_sources; ++i)
        {
            const float2 t = jones_station[i];
            const float val = (t.x * t.x + t.y * t.y) * source_I[i];
            OSKAR_KAHAN_SUM(float, sum, val, guard)
        }
        vis[s].x += sum;
    }
}

/* Double precision. */
void oskar_auto_correlate_scalar_omp_d(const int num_sources,
        const int num_stations, const double2* jones, const double* source_I,
        double2* vis)
{
    int i, s;
#pragma omp parallel for private(i, s)
    for (s = 0; s < num_stations; ++s)
    {
        double sum = 0.0;
        const double2 *const jones_station = &jones[s * num_sources];
        for (i = 0; i < num_sources; ++i)
        {
            const double2 t = jones_station[i];
            sum += (t.x * t.x + t.y * t.y) * source_I[i];
        }
        vis[s].x += sum;
    }
}

#ifdef __cplusplus
}
#endif
