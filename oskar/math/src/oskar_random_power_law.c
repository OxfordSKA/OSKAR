/*
 * Copyright (c) 2012-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_random_power_law.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

double oskar_random_power_law(double min, double max, double index)
{
    double b0 = 0.0, b1 = 0.0;
    if ((min <= 0.0) || (max < min))
    {
        return 0.0;
    }
    /* NOLINTNEXTLINE: We can use rand() here without concern. */
    const double r = (double)rand() / ((double)RAND_MAX + 1.0);
    if (fabs(index + 1.0) < DBL_EPSILON)
    {
        b0 = log(max);
        b1 = log(min);
        return exp((b0 - b1) * r + b1);
    }
    else
    {
        b0 = pow(min, index + 1.0);
        b1 = pow(max, index + 1.0);
        return pow(((b1 - b0) * r + b0), (1.0 / (index + 1.0)));
    }
}

#ifdef __cplusplus
}
#endif
