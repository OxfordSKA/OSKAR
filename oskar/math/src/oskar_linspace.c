/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_linspace.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_linspace_d(double* values, double a, double b, int n)
{
    int i = 0;
    double inc = 0.0;
    if (n <= 1)
    {
        values[0] = (a + b) / 2;
        return;
    }
    inc = (b - a) / (n - 1);
    for (i = 0; i < n; ++i)
    {
        values[i] = a + (i * inc);
    }
}

void oskar_linspace_f(float* values, float a, float b, int n)
{
    int i = 0;
    float inc = 0.0f;
    if (n <= 1)
    {
        values[0] = (a + b) / 2;
        return;
    }
    inc = (b - a) / (n - 1);
    for (i = 0; i < n; ++i)
    {
        values[i] = a + (i * inc);
    }
}

#ifdef __cplusplus
}
#endif
