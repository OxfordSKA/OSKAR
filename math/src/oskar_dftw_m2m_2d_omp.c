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

#include <oskar_dftw_m2m_2d_omp.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_dftw_m2m_2d_omp_f(const int n_in, const float wavenumber,
        const float* x_in, const float* y_in, const float2* weights_in,
        const int n_out, const float* x_out, const float* y_out,
        const float4c* data, float4c* output)
{
    int i_out = 0;

    /* Loop over output points. */
    #pragma omp parallel for private(i_out)
    for (i_out = 0; i_out < n_out; ++i_out)
    {
        int i;
        float xp_out, yp_out;
        float4c out;

        /* Clear output value. */
        out.a.x = 0.0f;
        out.a.y = 0.0f;
        out.b.x = 0.0f;
        out.b.y = 0.0f;
        out.c.x = 0.0f;
        out.c.y = 0.0f;
        out.d.x = 0.0f;
        out.d.y = 0.0f;

        /* Get the output position. */
        xp_out = wavenumber * x_out[i_out];
        yp_out = wavenumber * y_out[i_out];

        /* Loop over input points. */
        for (i = 0; i < n_in; ++i)
        {
            float2 weight;

            /* Calculate the DFT phase for the output position. */
            {
                float t;
                float2 w;

                /* Phase. */
                t = xp_out * x_in[i] + yp_out * y_in[i];
                weight.x = cosf(t);
                weight.y = sinf(t);

                /* Multiply the supplied DFT weight by the computed phase. */
                w = weights_in[i];
                t = weight.x; /* Copy the real part. */
                weight.x *= w.x;
                weight.x -= w.y * weight.y;
                weight.y *= w.x;
                weight.y += w.y * t;
            }

            /* Complex multiply-accumulate input signal and weight. */
            {
                float4c in;
                in = data[i * n_out + i_out];
                out.a.x += in.a.x * weight.x;
                out.a.x -= in.a.y * weight.y;
                out.a.y += in.a.y * weight.x;
                out.a.y += in.a.x * weight.y;
                out.b.x += in.b.x * weight.x;
                out.b.x -= in.b.y * weight.y;
                out.b.y += in.b.y * weight.x;
                out.b.y += in.b.x * weight.y;
                out.c.x += in.c.x * weight.x;
                out.c.x -= in.c.y * weight.y;
                out.c.y += in.c.y * weight.x;
                out.c.y += in.c.x * weight.y;
                out.d.x += in.d.x * weight.x;
                out.d.x -= in.d.y * weight.y;
                out.d.y += in.d.y * weight.x;
                out.d.y += in.d.x * weight.y;
            }
        }

        /* Store the output point. */
        output[i_out] = out;
    }
}

/* Double precision. */
void oskar_dftw_m2m_2d_omp_d(const int n_in, const double wavenumber,
        const double* x_in, const double* y_in, const double2* weights_in,
        const int n_out, const double* x_out, const double* y_out,
        const double4c* data, double4c* output)
{
    int i_out = 0;

    /* Loop over output points. */
    #pragma omp parallel for private(i_out)
    for (i_out = 0; i_out < n_out; ++i_out)
    {
        int i;
        double xp_out, yp_out;
        double4c out;

        /* Clear output value. */
        out.a.x = 0.0;
        out.a.y = 0.0;
        out.b.x = 0.0;
        out.b.y = 0.0;
        out.c.x = 0.0;
        out.c.y = 0.0;
        out.d.x = 0.0;
        out.d.y = 0.0;

        /* Get the output position. */
        xp_out = wavenumber * x_out[i_out];
        yp_out = wavenumber * y_out[i_out];

        /* Loop over input points. */
        for (i = 0; i < n_in; ++i)
        {
            double2 weight;

            /* Calculate the DFT phase for the output position. */
            {
                double t;
                double2 w;

                /* Phase. */
                t = xp_out * x_in[i] + yp_out * y_in[i];
                weight.x = cos(t);
                weight.y = sin(t);

                /* Multiply the supplied DFT weight by the computed phase. */
                w = weights_in[i];
                t = weight.x; /* Copy the real part. */
                weight.x *= w.x;
                weight.x -= w.y * weight.y;
                weight.y *= w.x;
                weight.y += w.y * t;
            }

            /* Complex multiply-accumulate input signal and weight. */
            {
                double4c in;
                in = data[i * n_out + i_out];
                out.a.x += in.a.x * weight.x;
                out.a.x -= in.a.y * weight.y;
                out.a.y += in.a.y * weight.x;
                out.a.y += in.a.x * weight.y;
                out.b.x += in.b.x * weight.x;
                out.b.x -= in.b.y * weight.y;
                out.b.y += in.b.y * weight.x;
                out.b.y += in.b.x * weight.y;
                out.c.x += in.c.x * weight.x;
                out.c.x -= in.c.y * weight.y;
                out.c.y += in.c.y * weight.x;
                out.c.y += in.c.x * weight.y;
                out.d.x += in.d.x * weight.x;
                out.d.x -= in.d.y * weight.y;
                out.d.y += in.d.y * weight.x;
                out.d.y += in.d.x * weight.y;
            }
        }

        /* Store the output point. */
        output[i_out] = out;
    }
}

#ifdef __cplusplus
}
#endif
