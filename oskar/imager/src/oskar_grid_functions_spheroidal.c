/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/oskar_grid_functions_spheroidal.h"

#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_grid_convolution_function_spheroidal(const int support,
        const int oversample, double* fn)
{
    int i = 0;
    const int gcf_size = oversample * (support + 1);
    const int extent = support * oversample;
    for (i = 0; i < gcf_size; ++i)
    {
        const double nu = (double)i / (double)extent;
        fn[i] = (1.0 - nu*nu) * oskar_grid_function_spheroidal(nu);
    }
}


void oskar_grid_correction_function_spheroidal(const int image_size,
        const int padding_gcf, double* fn)
{
    int i = 0;
    double inc = 0.0;
    const int extent = image_size / 2;
    if (padding_gcf > 0)
    {
        inc = 1.0 / (image_size * padding_gcf);
    }
    for (i = 0; i < image_size; ++i)
    {
        double val = 0.0, sinc_cor = 1.0;
        const double nu = (double)(i - extent) / (double)extent;
        if (padding_gcf > 0 && i != extent)
        {
            const double x = M_PI * (i - extent) * inc;
            sinc_cor = sin(x) / x;
        }
        val = oskar_grid_function_spheroidal(fabs(nu)) * (sinc_cor * sinc_cor);
        fn[i] = (val != 0.0) ? 1.0 / val : 1.0;
    }
}


/* Translated from FORTRAN routine in casacore/scimath_f/grdsf.f */
double oskar_grid_function_spheroidal(const double nu)
{
    int i = 0, row = 0, sp = 0, sq = 0;
    double numer = 0.0, denom = 0.0, end = 0.0, delta = 0.0;
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
