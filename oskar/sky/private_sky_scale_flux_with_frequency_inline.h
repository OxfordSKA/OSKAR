/*
 * Copyright (c) 2014, The University of Oxford
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

#include <oskar_global.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define C0  299792458.0
#define C0f 299792458.0f

/* Single precision. */
OSKAR_INLINE
void oskar_scale_flux_with_frequency_inline_f(const float frequency,
        float* I, float* Q, float* U, float* V, float* ref_freq,
        const float sp_index, const float rm)
{
    float freq0_, sin_b, cos_b;

    /* Get reference frequency, spectral index and rotation measure. */
    freq0_ = *ref_freq;
    if (freq0_ == 0.0f)
        return;

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
            b = 2.0f * rm * delta_lambda_squared;
#ifdef __CUDACC__
            sincosf(b, &sin_b, &cos_b);
#else
            sin_b = sinf(b);
            cos_b = cosf(b);
#endif
        }
    }

    /* Set new values and update reference frequency. */
    {
        float scale, Q_, U_;

        /* Compute spectral index scaling factor. */
        scale = powf(frequency / freq0_, sp_index);
        Q_ = scale * *Q;
        U_ = scale * *U;
        *I *= scale;
        *V *= scale;
        *Q = Q_ * cos_b - U_ * sin_b;
        *U = Q_ * sin_b + U_ * cos_b;
        *ref_freq = frequency;
    }
}

/* Double precision. */
OSKAR_INLINE
void oskar_scale_flux_with_frequency_inline_d(const double frequency,
        double* I, double* Q, double* U, double* V, double* ref_freq,
        const double sp_index, const double rm)
{
    double freq0_, sin_b, cos_b;

    /* Get reference frequency, spectral index and rotation measure. */
    freq0_ = *ref_freq;
    if (freq0_ == 0.0)
        return;

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
            b = 2.0 * rm * delta_lambda_squared;
#ifdef __CUDACC__
            sincos(b, &sin_b, &cos_b);
#else
            sin_b = sin(b);
            cos_b = cos(b);
#endif
        }
    }

    /* Set new values and update reference frequency. */
    {
        double scale, Q_, U_;

        /* Compute spectral index scaling factor. */
        scale = pow(frequency / freq0_, sp_index);
        Q_ = scale * *Q;
        U_ = scale * *U;
        *I *= scale;
        *V *= scale;
        *Q = Q_ * cos_b - U_ * sin_b;
        *U = Q_ * sin_b + U_ * cos_b;
        *ref_freq = frequency;
    }
}

#ifdef __cplusplus
}
#endif

