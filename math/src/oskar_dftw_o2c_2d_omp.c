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

#include <oskar_dftw_o2c_2d_omp.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#if 0
/* Single precision. */
void oskar_dftw_o2c_2d_omp_f(const int n_in, const float wavenumber,
        const float* x_in, const float* y_in, const float2* weights_in,
        const int n_out, const float* x_out, const float* y_out,
        float2* output)
{
    int i_out = 0;

    /* Loop over output points. */
    #pragma omp parallel for private(i_out)
    for (i_out = 0; i_out < n_out; ++i_out)
    {
        int i;
        float xp_out, yp_out;
        float2 out, out_c, out_cnt, out_new;

        /* Clear output value. */
        out.x = 0.0f;
        out.y = 0.0f;
        out_c.x = 0.0f;
        out_c.y = 0.0f;

        /* Get the output position. */
        xp_out = wavenumber * x_out[i_out];
        yp_out = wavenumber * y_out[i_out];

        /* Loop over input points. */
        for (i = 0; i < n_in; ++i)
        {
            float signal_x, signal_y;

            /* Calculate the phase for the output position. */
            {
                float a;
                a = xp_out * x_in[i] + yp_out * y_in[i];
                signal_x = cos(a);
                signal_y = sin(a);
            }

            /* Perform complex multiply-accumulate using Kahan summation. */
            {
                float2 w, r;
                w = weights_in[i];
                r.x = signal_x * w.x - signal_y * w.y;
                r.y = signal_x * w.y + signal_y * w.x;
                out_cnt.x = r.x - out_c.x;
                out_new.x = out.x + out_cnt.x;
                out_c.x = (out_new.x - out.x) - out_cnt.x;
                out.x = out_new.x;
                out_cnt.y = r.y - out_c.y;
                out_new.y = out.y + out_cnt.y;
                out_c.y = (out_new.y - out.y) - out_cnt.y;
                out.y = out_new.y;
            }
        }

        /* Store the output point. */
        output[i_out] = out;
    }
}
#endif

/* Single precision. */
void oskar_dftw_o2c_2d_omp_f(const int n_in, const float wavenumber,
        const float* x_in, const float* y_in, const float2* weights_in,
        const int n_out, const float* x_out, const float* y_out,
        float2* output)
{
    int i_out = 0;

    /* Loop over output points. */
    #pragma omp parallel for private(i_out)
    for (i_out = 0; i_out < n_out; ++i_out)
    {
        int i;
        float xp_out, yp_out;
        float2 out;

        /* Clear output value. */
        out.x = 0.0f;
        out.y = 0.0f;

        /* Get the output position. */
        xp_out = wavenumber * x_out[i_out];
        yp_out = wavenumber * y_out[i_out];

        /* Loop over input points. */
        for (i = 0; i < n_in; ++i)
        {
            float signal_x, signal_y;

            /* Calculate the phase for the output position. */
            {
                float a;
                a = xp_out * x_in[i] + yp_out * y_in[i];
                signal_x = cosf(a);
                signal_y = sinf(a);
            }

            /* Perform complex multiply-accumulate. */
            {
                float2 w;
                w = weights_in[i];
                out.x += signal_x * w.x;
                out.x -= signal_y * w.y;
                out.y += signal_y * w.x;
                out.y += signal_x * w.y;
            }
        }

        /* Store the output point. */
        output[i_out] = out;
    }
}

/* Double precision. */
void oskar_dftw_o2c_2d_omp_d(const int n_in, const double wavenumber,
        const double* x_in, const double* y_in, const double2* weights_in,
        const int n_out, const double* x_out, const double* y_out,
        double2* output)
{
    int i_out = 0;

    /* Loop over output points. */
    #pragma omp parallel for private(i_out)
    for (i_out = 0; i_out < n_out; ++i_out)
    {
        int i;
        double xp_out, yp_out;
        double2 out;

        /* Clear output value. */
        out.x = 0.0;
        out.y = 0.0;

        /* Get the output position. */
        xp_out = wavenumber * x_out[i_out];
        yp_out = wavenumber * y_out[i_out];

        /* Loop over input points. */
        for (i = 0; i < n_in; ++i)
        {
            double signal_x, signal_y;

            /* Calculate the phase for the output position. */
            {
                double a;
                a = xp_out * x_in[i] + yp_out * y_in[i];
                signal_x = cos(a);
                signal_y = sin(a);
            }

            /* Perform complex multiply-accumulate. */
            {
                double2 w;
                w = weights_in[i];
                out.x += signal_x * w.x;
                out.x -= signal_y * w.y;
                out.y += signal_y * w.x;
                out.y += signal_x * w.y;
            }
        }

        /* Store the output point. */
        output[i_out] = out;
    }
}

#ifdef __cplusplus
}
#endif
