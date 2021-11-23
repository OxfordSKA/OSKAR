/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_meshgrid.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_meshgrid_d(double* X, double* Y, const double* x, int nx,
        const double* y, int ny)
{
    int j = 0, i = 0;
    for (j = 0; j < ny; ++j)
    {
        for (i = 0; i < nx; ++i)
        {
            X[j * nx + i] = x[i];
            Y[j * nx + i] = y[ny - 1 - j];
        }
    }
}

void oskar_meshgrid_f(float* X, float* Y, const float* x, int nx,
        const float* y, int ny)
{
    int j = 0, i = 0;
    for (j = 0; j < ny; ++j)
    {
        for (i = 0; i < nx; ++i)
        {
            X[j * nx + i] = x[i];
            Y[j * nx + i] = y[ny - 1 - j];
        }
    }
}

#ifdef __cplusplus
}
#endif
