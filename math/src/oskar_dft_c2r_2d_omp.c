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

#include <oskar_dft_c2r_2d_omp.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_dft_c2r_2d_omp_f(const int num_in, const float wavenumber,
        const float* x_in, const float* y_in, const float2* data_in,
        const int num_out, const float* x_out, const float* y_out,
        float* output)
{
    int i_out = 0;

    /* Loop over output points. */
    #pragma omp parallel for private(i_out)
    for (i_out = 0; i_out < num_out; ++i_out)
    {
        int i;
        float xp_out, yp_out, out = 0.0f; /* Clear output value. */

        /* Get the output position. */
        xp_out = wavenumber * x_out[i_out];
        yp_out = wavenumber * y_out[i_out];

        /* Loop over input points. */
        for (i = 0; i < num_in; ++i)
        {
            /* Calculate the complex DFT weight. */
            float a, weight_x, weight_y;
            a = -(x_in[i] * xp_out + y_in[i] * yp_out);
            weight_x = cosf(a);
            weight_y = sinf(a);

            /* Perform complex multiply-accumulate.
             * Output is real, so only evaluate the real part. */
            out += data_in[i].x * weight_x; /* RE*RE */
            out -= data_in[i].y * weight_y; /* IM*IM */
        }

        /* Store the output point. */
        output[i_out] = out;
    }
}

/* Double precision. */
void oskar_dft_c2r_2d_omp_d(const int num_in, const double wavenumber,
        const double* x_in, const double* y_in, const double2* data_in,
        const int num_out, const double* x_out, const double* y_out,
        double* output)
{
    int i_out = 0;

    /* Loop over output points. */
    #pragma omp parallel for private(i_out)
    for (i_out = 0; i_out < num_out; ++i_out)
    {
        int i;
        double xp_out, yp_out, out = 0.0; /* Clear output value. */

        /* Get the output position. */
        xp_out = wavenumber * x_out[i_out];
        yp_out = wavenumber * y_out[i_out];

        /* Loop over input points. */
        for (i = 0; i < num_in; ++i)
        {
            /* Calculate the complex DFT weight. */
            double a, weight_x, weight_y;
            a = -(x_in[i] * xp_out + y_in[i] * yp_out);
            weight_x = cos(a);
            weight_y = sin(a);

            /* Perform complex multiply-accumulate.
             * Output is real, so only evaluate the real part. */
            out += data_in[i].x * weight_x; /* RE*RE */
            out -= data_in[i].y * weight_y; /* IM*IM */
        }

        /* Store the output point. */
        output[i_out] = out;
    }
}

#ifdef __cplusplus
}
#endif
