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

#include "math/oskar_dftw_c2c_3d_omp.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_dftw_c2c_3d_omp_f(const int n_in, const float wavenumber,
        const float* x_in, const float* y_in, const float* z_in,
        const float2* weights_in, const int n_out, const float* x_out,
        const float* y_out, const float* z_out, const float2* data,
        float2* output)
{
    int i_out = 0;

    /* Loop over output points. */
    #pragma omp parallel for private(i_out)
    for (i_out = 0; i_out < n_out; ++i_out)
    {
        int i;
        float xp_out, yp_out, zp_out;
        float2 out;

        /* Clear output value. */
        out.x = 0.0f;
        out.y = 0.0f;

        /* Get the output position. */
        xp_out = wavenumber * x_out[i_out];
        yp_out = wavenumber * y_out[i_out];
        zp_out = wavenumber * z_out[i_out];

        /* Loop over input points. */
        for (i = 0; i < n_in; ++i)
        {
            float2 temp, w;
            float a;

            /* Calculate the phase for the output position. */
            a = xp_out * x_in[i] + yp_out * y_in[i] + zp_out * z_in[i];
            temp.x = cosf(a);
            temp.y = sinf(a);

            /* Multiply the supplied DFT weight by the computed phase. */
            w = weights_in[i];
            a = w.x;
            w.x *= temp.x;
            w.x -= w.y * temp.y;
            w.y *= temp.x;
            w.y += a * temp.y;

            /* Perform complex multiply-accumulate. */
            temp = data[i * n_out + i_out];
            out.x += w.x * temp.x;
            out.x -= w.y * temp.y;
            out.y += w.y * temp.x;
            out.y += w.x * temp.y;
        }

        /* Store the output point. */
        output[i_out] = out;
    }
}

/* Double precision. */
void oskar_dftw_c2c_3d_omp_d(const int n_in, const double wavenumber,
        const double* x_in, const double* y_in, const double* z_in,
        const double2* weights_in, const int n_out, const double* x_out,
        const double* y_out, const double* z_out, const double2* data,
        double2* output)
{
    int i_out = 0;

    /* Loop over output points. */
    #pragma omp parallel for private(i_out)
    for (i_out = 0; i_out < n_out; ++i_out)
    {
        int i;
        double xp_out, yp_out, zp_out;
        double2 out;

        /* Clear output value. */
        out.x = 0.0;
        out.y = 0.0;

        /* Get the output position. */
        xp_out = wavenumber * x_out[i_out];
        yp_out = wavenumber * y_out[i_out];
        zp_out = wavenumber * z_out[i_out];

        /* Loop over input points. */
        for (i = 0; i < n_in; ++i)
        {
            double2 temp, w;
            double a;

            /* Calculate the phase for the output position. */
            a = xp_out * x_in[i] + yp_out * y_in[i] + zp_out * z_in[i];
            temp.x = cos(a);
            temp.y = sin(a);

            /* Multiply the supplied DFT weight by the computed phase. */
            w = weights_in[i];
            a = w.x;
            w.x *= temp.x;
            w.x -= w.y * temp.y;
            w.y *= temp.x;
            w.y += a * temp.y;

            /* Perform complex multiply-accumulate. */
            temp = data[i * n_out + i_out];
            out.x += w.x * temp.x;
            out.x -= w.y * temp.y;
            out.y += w.y * temp.x;
            out.y += w.x * temp.y;
        }

        /* Store the output point. */
        output[i_out] = out;
    }
}

#ifdef __cplusplus
}
#endif
