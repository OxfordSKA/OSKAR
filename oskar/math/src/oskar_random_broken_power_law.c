/*
 * Copyright (c) 2012-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_random_broken_power_law.h"
#include "math/oskar_random_power_law.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

double oskar_random_broken_power_law(double min, double max, double breakpoint,
        double index1, double index2)
{
    if ((min <= 0.0) || (max < min))
    {
        return 0.0;
    }

    if (min >= breakpoint)
    {
        return oskar_random_power_law(min, max, index1);
    }
    else if (max <= breakpoint)
    {
        return oskar_random_power_law(min, max, index2);
    }
    else
    {
        double b1 = 0.0, b2 = 0.0;
        const double index1p1 = index1 + 1.0;
        const double index2p1 = index2 + 1.0;

        if (fabs(index1p1) < DBL_EPSILON)
        {
            b1 = log(breakpoint / min);
        }
        else
        {
            b1 = (pow(breakpoint, index1p1) - pow(min, index1p1)) / index1p1;
        }

        if (fabs(index2p1) < DBL_EPSILON)
        {
            b2 = log(max / breakpoint) * pow(breakpoint, index1 - index2);
        }
        else
        {
            b2 = (pow(max, index2p1) - pow(breakpoint, index2p1)) *
                    pow(breakpoint, index1 - index2) / index2p1;
        }

        /* NOLINTNEXTLINE: We can use rand() here without concern. */
        const double r = (double)rand() / ((double)RAND_MAX + 1.0);
        if (r > b1 / (b1 + b2))
        {
            return oskar_random_power_law(breakpoint, max, index2);
        }
        else
        {
            return oskar_random_power_law(min, breakpoint, index1);
        }
    }
}

#ifdef __cplusplus
}
#endif
