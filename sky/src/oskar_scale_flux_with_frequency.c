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

#include <oskar_scale_flux_with_frequency.h>
#include <oskar_scale_flux_with_frequency_inline.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_scale_flux_with_frequency_f(int num_sources, float frequency,
        float* I, float* Q, float* U, float* V, float* ref_freq,
        const float* sp_index, const float* rm)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        oskar_scale_flux_with_frequency_inline_f(frequency,
                &I[i], &Q[i], &U[i], &V[i], &ref_freq[i], sp_index[i], rm[i]);
    }
}

/* Double precision. */
void oskar_scale_flux_with_frequency_d(int num_sources, double frequency,
        double* I, double* Q, double* U, double* V, double* ref_freq,
        const double* sp_index, const double* rm)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        oskar_scale_flux_with_frequency_inline_d(frequency,
                &I[i], &Q[i], &U[i], &V[i], &ref_freq[i], sp_index[i], rm[i]);
    }
}

#ifdef __cplusplus
}
#endif
