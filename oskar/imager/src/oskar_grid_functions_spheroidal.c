/*
 * Copyright (c) 2016, The University of Oxford
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

#include "imager/oskar_grid_functions_spheroidal.h"

#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_grid_convolution_function_spheroidal(const int support,
        const int oversample, double* fn)
{
    int i, extent, gcf_size;
    double nu;
    gcf_size = oversample * (support + 1);
    extent = support * oversample;
    for (i = 0; i < gcf_size; ++i)
    {
        nu = (double)i / (double)extent;
        fn[i] = (1.0 - nu*nu) * oskar_grid_function_spheroidal(nu);
    }
}


void oskar_grid_correction_function_spheroidal(const int image_size,
        const int padding_gcf, double* fn)
{
    int i, extent;
    double nu, val, inc = 0.0, sinc_cor, x;
    extent = image_size / 2;
    if (padding_gcf > 0)
        inc = 1.0 / (image_size * padding_gcf);
    for (i = 0; i < image_size; ++i)
    {
        sinc_cor = 1.0;
        nu = (double)(i - extent) / (double)extent;
        if (padding_gcf > 0 && i != extent)
        {
            x = M_PI * (i - extent) * inc;
            sinc_cor = sin(x) / x;
        }
        val = oskar_grid_function_spheroidal(fabs(nu)) * (sinc_cor * sinc_cor);
        fn[i] = (val != 0.0) ? 1.0 / val : 1.0;
    }
}


/* Translated from FORTRAN routine in casacore/scimath_f/grdsf.f */
double oskar_grid_function_spheroidal(const double nu)
{
    int i, row, sp, sq;
    double numer, denom, end, delta;
    static const double p[] =
    {
        0.08203343, -0.3644705, 0.627866, -0.5335581, 0.2312756,
        0.004028559, -0.03697768, 0.1021332, -0.1201436, 0.06412774
    };
    static const double q[] =
    {
        1.0, 0.8212018, 0.2078043,
        1.0, 0.9599102, 0.2918724
    };

    if (nu >= 0.0 && nu < 0.75)
    {
       row = 0;
       end = 0.75;
    }
    else if (nu >= 0.75 && nu <= 1.0)
    {
        row = 1;
        end = 1.0;
    }
    else
    {
       return 0.0;
    }

    delta = nu * nu - end * end;
    sp = row * 5;
    sq = row * 3;
    numer = p[sp];
    denom = q[sq];
    for (i = 1; i < 5; ++i) numer += p[sp + i] * pow(delta, i);
    for (i = 1; i < 3; ++i) denom += q[sq + i] * pow(delta, i);
    return denom == 0.0 ? 0.0 : numer / denom;
}

#ifdef __cplusplus
}
#endif
