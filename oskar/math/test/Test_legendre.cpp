/*
 * Copyright (c) 2019-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "math/define_legendre_polynomial.h"
#include "math/oskar_cmath.h"

#include <cstdio>

TEST(legendre, test)
{
    float x = 0.2f;
    float y = sqrt((1 - x) * (1 + x));
    int l_max = 5;
    for (int l = 0; l <= l_max; ++l)
    {
        for (int m = 0; m <= l; ++m)
        {
            float val0 = 0.0f, val1 = 0.0f, val2 = 0.0f;
            OSKAR_LEGENDRE1(float, l, m, x, y, val0);
            OSKAR_LEGENDRE2(float, l, m, x, y, val0, val1, val2);
            printf("P_%d^%d(x) = % .4f, scal = % .4f, grad = % .4f\n",
                    l, m, val0, val1, val2);
        }
    }
}
