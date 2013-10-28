/*
 * Copyright (c) 2013, The University of Oxford
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
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define C0  299792458.0
#define C0f 299792458.0f

/* Single precision. */
void oskar_scale_flux_with_frequency_f(int num_sources, float frequency,
        float* I, float* Q, float* U, float* V, float* ref_freq,
        const float* sp_index, const float* rm)
{
    int i;

    /* Loop over sources. */
    for (i = 0; i < num_sources; ++i)
    {
        float freq0_, spix_, rm_, sin_b, cos_b;

        /* Get reference frequency, spectral index and rotation measure. */
        freq0_ = ref_freq[i];
        spix_ = sp_index[i];
        rm_ = rm[i];

        /* Compute rotation factors, sin(2 beta) and cos(2 beta). */
        {
            float delta_lambda_squared;

            /* Compute delta_lambda_squared using factorised difference of two
             * squares. (Numerically superior than an explicit difference.)
             * This is (lambda^2 - lambda0^2) */
            {
                float lambda, lambda0;
                lambda  = C0f / frequency;
                lambda0 = C0f / freq0_;
                delta_lambda_squared = (lambda - lambda0) * (lambda + lambda0);
            }

            /* Compute sin(2 beta) and cos(2 beta). */
            {
                float b;
                b = 2.0f * rm_ * delta_lambda_squared;
                sin_b = sinf(b);
                cos_b = cosf(b);
            }
        }

        /* Set new values and update reference frequency. */
        {
            float scale, Q_, U_;

            /* Compute spectral index scaling factor. */
            scale = powf(frequency / freq0_, spix_);
            Q_ = scale * Q[i];
            U_ = scale * U[i];
            I[i] *= scale;
            V[i] *= scale;
            Q[i] = Q_ * cos_b - U_ * sin_b;
            U[i] = Q_ * sin_b + U_ * cos_b;
            ref_freq[i] = frequency;
        }
    }
}

/* Double precision. */
void oskar_scale_flux_with_frequency_d(int num_sources, double frequency,
        double* I, double* Q, double* U, double* V, double* ref_freq,
        const double* sp_index, const double* rm)
{
    int i;

    /* Loop over sources. */
    for (i = 0; i < num_sources; ++i)
    {
        double freq0_, spix_, rm_, sin_b, cos_b;

        /* Get reference frequency, spectral index and rotation measure. */
        freq0_ = ref_freq[i];
        spix_ = sp_index[i];
        rm_ = rm[i];

        /* Compute rotation factors, sin(2 beta) and cos(2 beta). */
        {
            double delta_lambda_squared;

            /* Compute delta_lambda_squared using factorised difference of two
             * squares. (Numerically superior than an explicit difference.)
             * This is (lambda^2 - lambda0^2) */
            {
                double lambda, lambda0;
                lambda  = C0 / frequency;
                lambda0 = C0 / freq0_;
                delta_lambda_squared = (lambda - lambda0) * (lambda + lambda0);
            }

            /* Compute sin(2 beta) and cos(2 beta). */
            {
                double b;
                b = 2.0 * rm_ * delta_lambda_squared;
                sin_b = sin(b);
                cos_b = cos(b);
            }
        }

        /* Set new values and update reference frequency. */
        {
            double scale, Q_, U_;

            /* Compute spectral index scaling factor. */
            scale = pow(frequency / freq0_, spix_);
            Q_ = scale * Q[i];
            U_ = scale * U[i];
            I[i] *= scale;
            V[i] *= scale;
            Q[i] = Q_ * cos_b - U_ * sin_b;
            U[i] = Q_ * sin_b + U_ * cos_b;
            ref_freq[i] = frequency;
        }
    }
}

#ifdef __cplusplus
}
#endif
